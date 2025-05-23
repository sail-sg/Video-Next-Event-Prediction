export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

task='mvbench'
output_path="./logs_mvbench/"

accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained=/path/to/model/,max_num_frames=32,use_flash_attention_2=True \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --verbosity=DEBUG \
        --log_samples_suffix 'debug' \
        --output_path $output_path
