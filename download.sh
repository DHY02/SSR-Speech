export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download \
        --local-dir ./pretrained_models/SSR-Speech-English \
        --repo-type model \
        westbrook/SSR-Speech-English
