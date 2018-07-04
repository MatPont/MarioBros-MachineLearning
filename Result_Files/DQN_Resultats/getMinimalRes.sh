echo -n Password:
read -s password

sshpass -p "$password" scp mpont@osirim-slurm.irit.fr:./mario-DQN/$1/src/main/bin/AmiCoBuild/JavaPy/episode_values.txt 1_episode_values.txt
sshpass -p "$password" scp mpont@osirim-slurm.irit.fr:./mario-DQN/$1_Iter1/src/main/bin/AmiCoBuild/JavaPy/episode_values.txt 2_episode_values.txt
sshpass -p "$password" scp mpont@osirim-slurm.irit.fr:./mario-DQN/$1_Iter2/src/main/bin/AmiCoBuild/JavaPy/episode_values.txt 3_episode_values.txt
sshpass -p "$password" scp mpont@osirim-slurm.irit.fr:./mario-DQN/$1_Iter3/src/main/bin/AmiCoBuild/JavaPy/episode_values.txt 4_episode_values.txt

sshpass -p "$password" scp mpont@osirim-slurm.irit.fr:./mario-DQN/$1/src/main/bin/AmiCoBuild/JavaPy/MarioDQNAgent2.py MarioDQNAgent2.py

