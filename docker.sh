docker run -d --name  polarlabelnb -e JUPYTER_ENABLE_LAB=yes -e PASSWORD="arkforn4f" -v $PWD/config:/home/jovyan/.jupyter/ -v $PWD:/home/jovyan/work/ -p 80:8888 -e GRANT_SUDO=yes --user root jupyter/datascience-notebook
chmod 777 -R  data/demo/stage/result/
echo '..Set password for web access'
#sleep 5
docker exec -it polarlabelnb jupyter notebook password
echo '..starting '
chmod 777 -R  config
docker restart polarlabelnb
echo 'Please navigate your browser to http://localhost/lab/tree/work/notebooks/run_experiment.ipynb'
