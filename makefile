CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`
ALL:
	g++ rhythm_folder.cpp $(CFLAGS) $(LIBS) -o rhythm_folder -std=c++17 -lstdc++fs -Ofast -I/share -DFMT_HEADER_ONLY
slow:
	g++ rhythm_folder.cpp $(CFLAGS) $(LIBS) -o rhythm_folder -std=c++17 -lstdc++fs -I/share
debug:
	g++ rhythm_folder.cpp $(CFLAGS) $(LIBS) -o rhythm_folder -std=c++17 -lstdc++fs -g -I/share
