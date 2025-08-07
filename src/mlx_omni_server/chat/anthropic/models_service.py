import logging
from datetime import datetime, timezone
from typing import Optional

from mlx_omni_server.chat.openai.models import ModelCacheScanner

from .schema import AnthropicModelInfo, AnthropicModelList


class AnthropicModelsService:
    def __init__(self):
        self.scanner = ModelCacheScanner()

    def list_models(
        self,
        limit: int = 20,
        after_id: Optional[str] = None,
        before_id: Optional[str] = None,
    ) -> AnthropicModelList:
        """List all available models in Anthropic format."""
        try:
            # Fetch all supported models and sort by most recently modified
            all_models_info = self.scanner.find_models_in_cache()
            all_models_info.sort(key=lambda x: x[0].last_modified, reverse=True)
        except Exception as e:
            logging.error(f"Error scanning cache for models: {str(e)}")
            all_models_info = []

        all_models = [
            AnthropicModelInfo(
                id=repo_info.repo_id,
                display_name=repo_info.repo_id,
                created_at=datetime.fromtimestamp(
                    repo_info.last_modified, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            for repo_info, _ in all_models_info
        ]

        # Basic pagination logic
        start_index = 0
        if after_id:
            try:
                start_index = (
                    next(i for i, m in enumerate(all_models) if m.id == after_id) + 1
                )
            except StopIteration:
                # after_id not found, return empty list
                start_index = len(all_models)

        # before_id is not fully supported in this basic implementation,
        # but we can prevent returning items after it.
        end_index = len(all_models)
        if before_id:
            try:
                end_index = next(
                    i for i, m in enumerate(all_models) if m.id == before_id
                )
            except StopIteration:
                pass

        paginated_data = all_models[start_index:end_index][:limit]

        has_more = (start_index + len(paginated_data)) < end_index

        return AnthropicModelList(
            data=paginated_data,
            first_id=paginated_data[0].id if paginated_data else None,
            last_id=paginated_data[-1].id if paginated_data else None,
            has_more=has_more,
        )
