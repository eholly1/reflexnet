# reflexnet

```
alias DD="docker container rm -f reflexnet"
alias DU="cd ~/reflexnet/docker && docker build --tag=reflexnetbuild . && docker run --name reflexnet -d reflexnet\
build"
alias DO="docker exec -it reflexnet bash"
```