#!/bin/bash
#SBATCH -p akya-cuda                                         # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A osivaz                                            # Kullanici adi
#SBATCH -J odir3500_swinV2_B384_SCA_SNDL_ASAM_IL             # Gonderilen isin ismi
#SBATCH -o odir3500_swinV2_B384_SCA_SNDL_ASAM_IL.out         # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:4                                         # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                                                 # Gorev kac node'da calisacak?
#SBATCH -n 1                                                 # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 40                                   # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=40:00:00                                      # Sure siniri koyun.


### Load modules
module load comp/python/ai-tools
module load apps/truba-ai/cpu-2024.0
module load apps/truba-ai/gpu-2024.0
echo "We have the modules: $(module list 2>&1)" > ${SLURM_JOB_ID}.info

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 odir3500_swinV2_B384_SCA_SNDL_ASAM_IL.py
exit