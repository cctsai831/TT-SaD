export PYTHONPATH=$PYTHONPATH:$(pwd)

MPI="-n 10 --mca btl ^openib --allow-run-as-root"

MODEL_FLAGS="--image_size XXX --num_channels XXX --num_res_blocks X --class_cond False --noise_schedule linear --diffusion_steps XXXX --timestep_respacing XXX --batch_size XX"

DIFFUSION_MODEL_PATH="--model_path path/to/diffusion/model"

STAIN_MATRIX="--stain_matrix_dir path/to/domain/center"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 mpiexec $MPI python3 scripts/image_sample_staindiff.py $MODEL_FLAGS $DIFFUSION_MODEL_PATH $STAIN_MATRIX \
   --base_samples path/to/data/to/be/adapted --save_dir path/to/which/output/