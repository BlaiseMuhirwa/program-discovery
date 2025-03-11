

mnist-bench-flatnav: 
	poetry run python evaluate.py \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16