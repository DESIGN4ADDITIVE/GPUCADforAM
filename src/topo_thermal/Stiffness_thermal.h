

#pragma once

void MakeRefStiff_t(REAL RefStiffone[8][8])
{

    RefStiffone[0][0] = 4.0; RefStiffone[0][1] = 0.0;  RefStiffone[0][3] = -1.0; RefStiffone[0][2] = 0.0;
    RefStiffone[0][4] = 0.0; RefStiffone[0][5] = -1.0; RefStiffone[0][6] = -1.0; RefStiffone[0][7] = -1.0;
    
    
    RefStiffone[1][0] = 0.0; RefStiffone[1][1] = 4.0;  RefStiffone[1][3] = 0.0; RefStiffone[1][2] = -1.0;
    RefStiffone[1][4] = -1.0; RefStiffone[1][5] = 0.0; RefStiffone[1][7] = -1.0; RefStiffone[1][6] = -1.0;
  
    RefStiffone[3][0] = -1.0; RefStiffone[3][1] = 0.0;  RefStiffone[3][3] = 4.0; RefStiffone[3][2] = 0.0;
    RefStiffone[3][4] = -1.0; RefStiffone[3][5] = -1.0; RefStiffone[3][7] = 0.0; RefStiffone[3][6] = -1.0;
  
    RefStiffone[2][0] = 0.0; RefStiffone[2][1] = -1.0;  RefStiffone[2][3] = 0.0; RefStiffone[2][2] = 4.0;
    RefStiffone[2][4] = -1.0; RefStiffone[2][5] = -1.0; RefStiffone[2][7] = -1.0; RefStiffone[2][6] = 0.0;

    RefStiffone[4][0] = 0.0; RefStiffone[4][1] = -1.0;  RefStiffone[4][3] = -1.0; RefStiffone[4][2] = -1.0;
    RefStiffone[4][4] = 4.0; RefStiffone[4][5] = 0.0; RefStiffone[4][7] = -1.0; RefStiffone[4][6] = 0.0;

    RefStiffone[5][0] = -1.0; RefStiffone[5][1] = 0.0;  RefStiffone[5][3] = -1.0; RefStiffone[5][2] = -1.0;
    RefStiffone[5][4] = 0.0; RefStiffone[5][5] = 4.0; RefStiffone[5][7] = 0.0; RefStiffone[5][6] = -1.0;
    
    RefStiffone[7][0] = -1.0; RefStiffone[7][1] = -1.0;  RefStiffone[7][3] = 0.0; RefStiffone[7][2] = -1.0;
    RefStiffone[7][4] = -1.0; RefStiffone[7][5] = 0.0; RefStiffone[7][7] = 4.0; RefStiffone[7][6] = 0.0;

    RefStiffone[6][0] = -1.0; RefStiffone[6][1] = -1.0;  RefStiffone[6][3] = -1.0; RefStiffone[6][2] = 0.0;
    RefStiffone[6][4] = 0.0; RefStiffone[6][5] = -1.0; RefStiffone[6][7] = 0.0; RefStiffone[6][6] = 4.0;

}


void MakeEleStiff_t(REAL k,REAL EleStiffone[8][8], REAL RefStiffone[8][8])
{
  #pragma omp parallel for
  for(int i=0;i<8;i++)
  {
    for(int j=0;j<8;j++)
    
    {
      REAL valll = (1.0/12.0)*k*RefStiffone[i][j];
      EleStiffone[i][j] = valll;
    }
    
  }
  
}