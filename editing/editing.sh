SAVE_PATH="/kuacc/users/aanees20/hpc_run/CLIPStyleGAN_eval_data/latent_editing/"

for i in 3
do
    echo $i
    python /kuacc/users/aanees20/hpc_run/DynaGAN/editing/inference_face_editing.py --out_path ${SAVE_PATH}/  --factor ${i}
done