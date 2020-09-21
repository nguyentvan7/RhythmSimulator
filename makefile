CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
ALL:
	g++ rhythm_folder.cpp $(CFLAGS) $(LIBS) -o rhythm_folder -std=c++17 -lstdc++fs
debug:
	g++ rhythm_folder.cpp $(CFLAGS) $(LIBS) -o rhythm_folder -std=c++17 -lstdc++fs -g
