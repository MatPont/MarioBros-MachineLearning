echo -n Password: 
read -s password
./getRes.sh $1 $password
./getRes.sh $1_Iter1 $password
./getRes.sh $1_Iter2 $password
./getRes.sh $1_Iter3 $password
