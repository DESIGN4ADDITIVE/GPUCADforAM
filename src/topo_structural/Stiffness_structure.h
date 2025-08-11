
#pragma once
#include "../general/topopt_defines.h"

REAL ShapeFunc(int i, REAL x, REAL y, REAL z)
{
  
  switch(i)
  {
    case 0:
	  return (1.0-x)*(1.0-y)*(1.0-z);
	case 1:
	  return x*(1.0-y)*(1.0-z);
	case 2:
	  return (1.0-x)*y*(1.0-z);
	case 3:
	  return x*y*(1.0-z);
	case 4:
	  return (1.0-x)*(1.0-y)*z;
	case 5:
	  return x*(1.0-y)*z;
	case 6:
	  return (1.0-x)*y*z;
	case 7:
	  return x*y*z;
	default:
	  cerr << "ERROR: ShapeFunc i out of range" << endl;
	  exit(1);
  }
  return 0.0;
}

//i: Number of the Shape Function
//j: Derivative direction
REAL GradShapeFunc(const unsigned int i, const unsigned int j, REAL x, REAL y, REAL z)
{
  switch(i)
  {
    case 0:
	  if(j==0) return -1.0*(1.0-y)*(1.0-z);
	  if(j==1) return -1.0*(1.0-x)*(1.0-z);
	  if(j==2) return -1.0*(1.0-x)*(1.0-y);
	  break;
	case 1:
	  if(j==0) return (1.0-y)*(1.0-z);
	  if(j==1) return -1.0*x*(1.0-z);
	  if(j==2) return -1.0*x*(1.0-y);
	  break;
	case 2:
	  if(j==0) return -1.0*y*(1.0-z);
	  if(j==1) return (1.0-x)*(1.0-z);
	  if(j==2) return -1.0*(1.0-x)*y;
	  break;
	case 3:
	  if(j==0) return y*(1.0-z);
	  if(j==1) return x*(1.0-z);
	  if(j==2) return -1.0*x*y;
	  break;
	case 4:
	  if(j==0) return -1.0*(1.0-y)*z;
	  if(j==1) return -1.0*(1.0-x)*z;
	  if(j==2) return (1.0-x)*(1.0-y);
	  break;
	case 5:
	  if(j==0) return (1.0-y)*z;
	  if(j==1) return -1.0*x*z;
	  if(j==2) return x*(1.0-y);
	  break;
	case 6:
	  if(j==0) return -1.0*y*z;
	  if(j==1) return (1.0-x)*z;
	  if(j==2) return (1.0-x)*y;
	  break;
	case 7:
	  if(j==0) return y*z;
	  if(j==1) return x*z;
	  if(j==2) return x*y;
	  break;
	default:
	  cerr << "ERROR: GradShapeFunc i out of range" << endl;
	  exit(1);
  }
  return 0.0;
}

void MakeRefStiff_s(REAL RefStiff[3][8][3][8])
{
  // i and j corresponds to [B] transpose while k and l corresponds to matrix [B]
  for(int i=0;i<3;i++)
  {
    for(int j=0;j<8;j++)
    {
      for(int k=0;k<3;k++)
      {
        for(int l=0;l<8;l++)
        {
          REAL Value = 0.0;
          const REAL a = 0.0;
          const REAL b = 1.0;
          const REAL fac1 = (b-a)/2.0;
          const REAL fac2 = (a+b)/2.0;

          ///intergating at each node of the neighbouring element in consideration 
          /// with respect to the thread node.
          for(int m=0;m<27;m++)
          {
            REAL xEval = fac1*EvalPos[m][0]+fac2;
            REAL yEval = fac1*EvalPos[m][1]+fac2;
            REAL zEval = fac1*EvalPos[m][2]+fac2;
            Value += weight[m]*GradShapeFunc(j, i, xEval, yEval, zEval)*GradShapeFunc(l, k, xEval, yEval, zEval);
         
          }
        

          RefStiff[i][j][k][l] = fac1*fac1*fac1*Value;

        }
      }
    }
  }
  
}

void MakeEleStiffness_s(REAL EleStiff[24][24], REAL RefStiff[3][8][3][8], const REAL E, const REAL nu)
{
  #pragma omp parallel for
  for(int i=0;i<24;i++)
  {
    for(int j=0;j<24;j++)
    {
      EleStiff[i][j] = 0.0;
    }
  }
  #pragma omp parallel for

  for(int i=0;i<8;i++)
  {
    for(int j=0;j<8;j++)
    {
      //first set of equations for u1
      EleStiff[i][j] += (1.0-nu)*RefStiff[0][i][0][j];
      EleStiff[i][j+8] += nu*RefStiff[0][i][1][j];
      EleStiff[i][j+16] += nu*RefStiff[0][i][2][j];

      EleStiff[i][j] += 0.5*(0.5-nu)*RefStiff[1][i][1][j];
      EleStiff[i][j+8] += 0.5*(0.5-nu)*RefStiff[1][i][0][j];

      EleStiff[i][j] += 0.5*(0.5-nu)*RefStiff[2][i][2][j];
      EleStiff[i][j+16] += 0.5*(0.5-nu)*RefStiff[2][i][0][j];


      //second set of equations for u2
      EleStiff[i+8][j] += nu*RefStiff[1][i][0][j];
      EleStiff[i+8][j+8] += (1.0-nu)*RefStiff[1][i][1][j];
      EleStiff[i+8][j+16] += nu*RefStiff[1][i][2][j];

      EleStiff[i+8][j] += 0.5*(0.5-nu)*RefStiff[0][i][1][j];
      EleStiff[i+8][j+8] += 0.5*(0.5-nu)*RefStiff[0][i][0][j];

      EleStiff[i+8][j+8] += 0.5*(0.5-nu)*RefStiff[2][i][2][j];
      EleStiff[i+8][j+16] += 0.5*(0.5-nu)*RefStiff[2][i][1][j];


      //third set of equations for u3
      EleStiff[i+16][j] += nu*RefStiff[2][i][0][j];
      EleStiff[i+16][j+8] += nu*RefStiff[2][i][1][j];
      EleStiff[i+16][j+16] += (1.0-nu)*RefStiff[2][i][2][j];

      EleStiff[i+16][j] += 0.5*(0.5-nu)*RefStiff[0][i][2][j];
      EleStiff[i+16][j+16] += 0.5*(0.5-nu)*RefStiff[0][i][0][j];

      EleStiff[i+16][j+8] += 0.5*(0.5-nu)*RefStiff[1][i][2][j];
      EleStiff[i+16][j+16] += 0.5*(0.5-nu)*RefStiff[1][i][1][j];
    }
  }
  //add the factor E/((1+nu)*(1-2nu)) to the element stiffness
  #pragma omp parallel for
  for(int i=0;i<24;i++)
  {
    for(int j=0;j<24;j++)
    {
      EleStiff[i][j] *= E/(2.0*(1.0+nu)*(1.0-2.0*nu));
     
    }
  }
}
