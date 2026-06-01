    source "$HOME/.exoenv/bin/activate"
    EXO_MODELS_READ_ONLY_DIRS="$HOME/.exo/lmstudio-links" \
    command exo &
    local exo_pid=$!

    # Wait for Exo API to become available
    echo -n "Waiting for Exo API"
    local retries=30
    while (( retries-- > 0 )); do
        curl -sf http://localhost:52415/v1/models >/dev/null 2>&1 && break
        echo -n "." 
        sleep 1
    done
    echo ""

    if ! curl -sf http://localhost:52415/v1/models >/dev/null 2>&1; then
        echo "Exo failed to start"
        kill $exo_pid 2>/dev/null
        source "$HOME/.venv/bin/activate"
        return 1
    fi  

    # Load model instance via placement API 
    echo "Loading model instance: $exo_model"
    local encoded_model="${exo_model//\//%2F}"
    local placement
    placement=$(curl -sf "http://localhost:52415/instance/placement?model_id=${encoded_model}")
    if [[ -z "$placement" ]]; then
        echo "Failed to get placement for $exo_model"
        kill $exo_pid 2>/dev/null
        source "$HOME/.venv/bin/activate"
        return 1
    fi  

    curl -sf -X POST http://localhost:52415/instance \
        -H "Content-Type: application/json" \
        -d "{\"instance\": ${placement}}" >/dev/null

    # Wait for instance to be ready
    echo -n "Waiting for model to load"
    retries=60
    while (( retries-- > 0 )); do
        local result
        result=$(curl -sf -X POST http://localhost:52415/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"${exo_model}\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false}" 2>/dev/null)
        if [[ "$result" != *"No instance found"* ]] && [[ "$result" != *"404"* ]] && [[ -n "$result" ]]; then
            echo " ready."
            break
        fi  
        echo -n "." 
        sleep 2
    done
    echo ""
    source "$HOME/Developer/mlx-omni-server/.venv/bin/activate"
    MLX_OMNI_EXO_URL=http://localhost:52415 \
    MLX_OMNI_STOP_WORDS="$stop_words" \
    MLX_OMNI_MODEL="$exo_model" \
    MLX_OMNI_TEMPERATURE="$exo_temp" \
    MLX_OMNI_TOP_K="$exo_topk" \
    MLX_OMNI_TOP_P="$exo_topp" \
    MLX_OMNI_MIN_P="$exo_minp" \
    MLX_OMNI_CONTEXT_SIZE="$exo_ctx" \
    MLX_OMNI_MAX_TOKENS=32000 \
    mlx-omni-server --port 8082

    kill $exo_pid 2>/dev/null
    source "$HOME/.venv/bin/activate"
