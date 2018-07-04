echo $1

mkdir $1
sshpass -p "$2" scp mpont@osirim-slurm.irit.fr:./mario-DQN/$1/src/main/bin/AmiCoBuild/JavaPy/* ./$1
