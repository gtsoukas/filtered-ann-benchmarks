# Weaviate setup

## Setup
```{bash}
git clone https://github.com/semi-technologies/weaviate.git
cd weaviate
docker build --target weaviate -t weaviate-latest .

docker-compose up -d
```

## Cleanup
```{bash}
docker-compose down
```
