#ifndef DATA_H
#define DATA_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "neural.h"

double* convert_csv_data(char* filename){

    double* data = (double*)malloc(sizeof(double) * 31541);

    FILE* fp = fopen(filename, "r");
    if (!fp){
        printf("Error opening file\n");
        exit(1);
    }
    else{
        char buffer[1024];
        int row = 0;
        int col = 0;
   
        double open = 0;
        double high = 0;
        double low = 0;
        double close = 0;

        while(fgets(buffer, 1024, fp)){
            col = 0;
            char*  value = strtok(buffer, ",");
            while(value){
                if (col == 2)
                    open = atof(value);
                else if (col == 3)
                    high = atof(value);
                else if (col == 4)
                    low = atof(value);
                else if (col == 5)
                    close = atof(value);
                value = strtok(NULL, ",");
                col++;
            }
            data[row] = 100 * (close - open) / open;
            row++;
        }
        fclose(fp);
    }
    return data;
}

void free_data(double* data){
    free(data);
}

#endif
