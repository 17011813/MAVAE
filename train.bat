@echo off
set b=2

python maml_vae.py --num=4 --batch_size=%b% --digits 5 6 7 8 9 10 11 12 13 14 15 16
python maml_vae.py --num=5 --batch_size=%b% --digits 4 6 7 8 9 10 11 12 13 14 15 16
python maml_vae.py --num=6 --batch_size=%b% --digits 4 5 7 8 9 10 11 12 13 14 15 16
python maml_vae.py --num=7 --batch_size=%b% --digits 4 5 6 8 9 10 11 12 13 14 15 16
python maml_vae.py --num=8 --batch_size=%b% --digits 4 5 6 7 9 10 11 12 13 14 15 16
python maml_vae.py --num=9 --batch_size=%b% --digits 4 5 6 7 8 10 11 12 13 14 15 16
python maml_vae.py --num=10 --batch_size=%b% --digits 4 5 6 7 8 9 11 12 13 14 15 16
python maml_vae.py --num=11 --batch_size=%b% --digits 4 5 6 7 8 9 10 12 13 14 15 16
python maml_vae.py --num=12 --batch_size=%b% --digits 4 5 6 7 8 9 10 11 13 14 15 16
python maml_vae.py --num=13 --batch_size=%b% --digits 4 5 6 7 8 9 10 11 12 14 15 16
python maml_vae.py --num=14 --batch_size=%b% --digits 4 5 6 7 8 9 10 11 12 13 15 16
python maml_vae.py --num=15 --batch_size=%b% --digits 4 5 6 7 8 9 10 11 12 13 14 16
python maml_vae.py --num=16 --batch_size=%b% --digits 4 5 6 7 8 9 10 11 12 13 14 15