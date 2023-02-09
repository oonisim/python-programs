//--------------------------------------------------------------------------------
// Echo command line arguments (excluding command itself)
//--------------------------------------------------------------------------------
#include <stdio.h>

int main(int argc, char * argv[], char * envp[]){
    if (argc == 1){
        return 0;
    }
    for (int i = 1; i < argc; i++){     // splint fails as https://stackoverflow.com/q/10257470/4281353
        int j = 0;
        while (argv[i][j] != '\0'){
            putchar(argv[i][j]);
            ++j;
        }
    }
    putchar('\n');
    return 0;
}
