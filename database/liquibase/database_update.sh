case_1(){
  liquibase --defaults-file=main.properties updateSQL
}

case_2(){
  liquibase --defaults-file=main.properties update
}

if [[ $1 -eq 1 ]]
then
  case_1
elif [[ $1 -eq 2 ]]
then
  case_2
else
  echo "Something is wrong"
fi
