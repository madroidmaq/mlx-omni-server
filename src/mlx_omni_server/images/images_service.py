import base64
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.error.exceptions import StopImageGenerationException
from mflux.flux.flux import Flux1
from PIL import Image

from ..utils.logger import logger
from .schema import ImageGenerationRequest, ImageObject, ResponseFormat


class MFluxImageGenerator:
    """Image generator using mflux library"""

    def __init__(self, model_version: str = "dhairyashil/FLUX.1-schnell-mflux-4bit"):
        self.model_version = model_version

        # Initialize model instance (lazy loading)
        self._flux = None

    def _extra_base_model(self, model_name: str):
        # List of supported base models
        supported_base_models = ["schnell", "dev", "dev-fill", "dev-depth", "dev-redux"]
        base_model = None
        # Extract base_model from model_name if it contains any of the supported keywords
        model_name_lower = model_name.lower()
        for base in supported_base_models:
            if base in model_name_lower:
                base_model = base
                logger.info(
                    f"Extracted base_model '{base_model}' from model_name '{model_name}'"
                )
                break

        # If we couldn't extract a base_model, set it to None
        if not base_model:
            logger.info(
                f"Could not extract base_model from model_name '{model_name}', using None"
            )

        return base_model

    def _get_flux(self, params: dict = None) -> Flux1:
        """Get or initialize Flux1 instance"""
        if self._flux is None:
            # Extract model name from full path
            model_name = self.model_version

            # Get base_model from params or extract from model_name
            base_model = params.get("base-model") if params else None

            # If base_model is not provided, try to extract it from model_name
            if model_name.__contains__("/") and not base_model:
                base_model = self._extra_base_model(model_name)

            # Let mflux handle model configuration
            self._flux = Flux1(
                model_config=ModelConfig.from_name(
                    model_name=model_name, base_model=base_model
                ),
                quantize=params.get("quantize"),
                local_path=params.get("local_path"),
                lora_paths=params.get("lora-paths") if params else None,
                lora_scales=params.get("lora-scales") if params else None,
            )

        return self._flux

    def _parse_size(self, size_str: str) -> Tuple[int, int]:
        """Parse size string to width and height"""
        try:
            width, height = map(int, size_str.split("x"))
            return width, height
        except (ValueError, AttributeError):
            return 1024, 1024

    def generate(
        self,
        request: ImageGenerationRequest,
        output_path: str,
        **extra_params,
    ) -> Image.Image:
        """Generate image using mflux"""
        # Parse image dimensions
        width, height = self._parse_size(request.size)

        # Get extra parameters from request
        request_extra_params = request.get_extra_params()

        # Merge all extra parameters, with passed extra_params taking precedence
        all_extra_params = {**request_extra_params, **extra_params}
        logger.info(f"all_extra_params: {all_extra_params}")

        # Generate random seed if not specified
        seed = all_extra_params.pop("seed", random.randint(0, 2**32 - 1))

        # Get or initialize Flux1 instance
        flux = self._get_flux(all_extra_params)

        # Generate image
        low_memory_mode = all_extra_params.get("low_arm", True)
        memory_saver = None
        if low_memory_mode:
            memory_saver = MemorySaver(flux=flux, keep_transformer=seed > 1)
            CallbackRegistry.register_before_loop(memory_saver)
            CallbackRegistry.register_in_loop(memory_saver)
            CallbackRegistry.register_after_loop(memory_saver)

        try:
            # Generate image
            image = flux.generate_image(
                seed=seed,
                prompt=request.prompt,
                config=Config(
                    num_inference_steps=all_extra_params.pop("steps", 4),
                    height=height,
                    width=width,
                    guidance=all_extra_params.pop("guidance", 4.0),
                ),
            )

            # Save image
            image.save(path=output_path, export_json_metadata=False)
            return image
        except StopImageGenerationException as e:
            raise Exception(f"Image generation interrupted: {str(e)}")
        except Exception as e:
            raise Exception(f"Error generating image: {str(e)}")
        finally:
            if memory_saver:
                print(memory_saver.memory_stats())


class ImagesService:
    def __init__(self):
        # Use system temporary directory
        self.output_dir = Path(tempfile.gettempdir()) / "mlx_omni_server" / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache loaded generator instances
        self._generator_cache: Dict[str, MFluxImageGenerator] = {}

    def _get_generator(self, model_name: str) -> MFluxImageGenerator:
        """Get or create image generator instance"""
        if model_name not in self._generator_cache:
            self._generator_cache[model_name] = MFluxImageGenerator(
                model_version=model_name
            )
        return self._generator_cache[model_name]

    def _get_output_path(self, uid: str) -> str:
        """Generate unique output path for image"""
        return str(self.output_dir / f"{uid}.png")

    def _image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _cleanup_image(self, image_path: str):
        """Clean up temporary image file"""
        try:
            os.unlink(image_path)
        except Exception as e:
            print(f"Error cleaning up image {image_path}: {str(e)}")

    def generate_images(
        self,
        request: ImageGenerationRequest,
    ) -> List[ImageObject]:
        """Generate images based on the request"""
        generated_images = []
        generator = self._get_generator(model_name=request.model)

        for i in range(request.n):
            # Generate unique identifier for this image
            uid = f"{int(time.time())}_{i}"
            output_path = self._get_output_path(uid)

            try:
                # Generate the image
                generator.generate(
                    request=request, output_path=output_path, low_memory_mode=True
                )

                # Create response object based on format
                image_object = ImageObject(revised_prompt=request.prompt)

                # Response format
                if request.response_format == ResponseFormat.B64_JSON:
                    image_object.b64_json = self._image_to_base64(output_path)
                else:  # URL format
                    image_object.url = f"file://{output_path}"

                generated_images.append(image_object)

            except Exception as e:
                raise Exception(f"Error generating image: {str(e)}")
            finally:
                # Clean up temporary file if using base64 format
                if request.response_format == ResponseFormat.B64_JSON:
                    self._cleanup_image(output_path)

        return generated_images
