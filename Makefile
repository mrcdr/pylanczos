pylanczos:
	g++ -O3 -Wall -shared -std=c++11 -fPIC -I lambda-lanczos/include/lambda_lanczos  `python3 -m pybind11 --includes` pylanczos.cpp -o pylanczoscpp`python3-config --extension-suffix`
