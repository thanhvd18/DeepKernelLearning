cd /Users/macbook/Documents/WorkSpace/DeepKernelLearning/experiments/scripts
export PYTHONPATH=/Users/macbook/Documents/WorkSpace/DeepKernelLearning:$PYTHONPATH


python tuning.py  --modality PET --version 3 \
            --kernel_type rbf \
            --epochs 1000 --dropout_rate 0.1 --patience 200 \
            --batch_size 64 --lr 1e-4 --lambda_ 0.75 \
            --hidden_dims 1000 60 \
            --latent_dim 30 

# python tuning.py  --modality PET --version 4 \
#             --kernel_type rbf \
#             --epochs 1000 --dropout_rate 0.1 --patience 200 \
#             --batch_size 64 --lr 1e-3 --lambda_ 0.7 \
#             --hidden_dims 1000 60 \
#             --latent_dim 30 

# python tuning.py  --modality PET --version 5 \
#             --kernel_type rbf \
#             --epochs 1000 --dropout_rate 0.1 --patience 200 \
#             --batch_size 64 --lr 1e-5 --lambda_ 0.7 \
#             --hidden_dims 1000 60 \
#             --latent_dim 30 

# python tuning.py  --modality PET --version 6 \
#             --kernel_type rbf \
#             --epochs 1000 --dropout_rate 0.1 --patience 200 \
#             --batch_size 32 --lr 1e-4 --lambda_ 0.7 \
#             --hidden_dims 1000 60\
#             --latent_dim 30 \

# python tuning.py  --modality PET --version 7 \
#             --kernel_type rbf \
#             --epochs 1000 --dropout_rate 0.1 --patience 200 \
#             --batch_size 64 --lr 1e-4 --lambda_ 0.75 \
#             --hidden_dims 1000 100\
#             --latent_dim 60 \

# python tuning.py  --modality PET --version 8 \
#             --kernel_type rbf \
#             --epochs 1000 --dropout_rate 0.1 --patience 200 \
#             --batch_size 64 --lr 1e-4 --lambda_ 0.75 \
#             --hidden_dims 1000 160\
#             --latent_dim 60 \

# python tuning.py  --modality PET --version 9 \
#             --kernel_type rbf \
#             --epochs 1000 --dropout_rate 0.1 --patience 200 \
#             --batch_size 64 --lr 1e-4 --lambda_ 0.75 \
#             --hidden_dims 1000 110\
#             --latent_dim 30 

