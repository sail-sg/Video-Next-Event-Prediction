
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

output_path="./logs_seedbench_r1/"
model_path="/path/to/model/"


task='seedbench_r1_l1'
accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained=${model_path},max_num_frames=32,use_flash_attention_2=True \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --verbosity=DEBUG \
        --log_samples_suffix 'debug' \
        --output_path $output_path


task='seedbench_r1_l2'
accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained=${model_path},max_num_frames=32,use_flash_attention_2=True \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --verbosity=DEBUG \
        --log_samples_suffix 'debug' \
        --output_path $output_path


task='seedbench_r1_l3'
accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained=${model_path},max_num_frames=32,use_flash_attention_2=True \
        --tasks $task \
        --batch_size 1 \
        --log_samples \
        --verbosity=DEBUG \
        --log_samples_suffix 'debug' \
        --output_path $output_path

