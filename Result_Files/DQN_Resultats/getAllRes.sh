echo -n Password:
read -s password

echo $1
mkdir $1
cp getRes.sh $1
cd $1

for iter in `seq 0 2`
do
	#mkdir $1/Iter$iter
	#cp getRes.sh $1/Iter$iter
	for i in `seq 0 1`
	do
		for j in `seq 0 1`
		do
			#cd ./$1/Iter$iter
			if [ $iter -eq 0 ]
			then
				./getRes.sh mario-DQN_S$i-R$j $password
			else
				./getRes.sh mario-DQN_S$i-R{$j}_Iter$iter $password
			fi
			#cd ..
		done	
	done
	#rm $1/Iter$iter/getRes.sh 
done

cd ..
rm $1/getRes.sh
