
bin/XS.so:
	@mkdir -p bin
	nvcc --compiler-options='-fPIC' -use_fast_math -lineinfo --ptxas-options=-v -shared -o bin/XS.so src/XS.cu src/vdW.cu src/WaasKirf.cu

clean:
	@echo " Cleaning ...";
	@rm bin/*
