#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <time.h>  /* clock_t, clock, CLOCKS_PER_SEC */
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Eigen/IterativeLinearSolvers"
#include <vector>
#include "iohpc.h"
#include "def.h"
using namespace Eigen;
using namespace std;

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double

/*
  2D Transport Solution With Step Characteristics and LOQD
  
  Programer : Luke Cornejo
  Version   : 1.0
  Date      : 2-19-14
  
                         Changes
  ******************************************************

  
  To complile
  Windows
c++ -I .\eigen SC_LOQDhpc1.0.cpp iotrans.cpp def.cpp -o SC_LOQDhpc1.0.exe -std=c++0x
  Linux
  
  hpc
c++ -I .\eigen SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0.exe
c++ -I .\eigen -O SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0gO.exe
c++ -I .\eigen -O2 SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0gO2.exe
c++ -I .\eigen -O3 SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0gO3.exe
c++ -I .\eigen -Os SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0gOs.exe
c++ -I .\eigen -O3 -ffast-math SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0gOf.exe

icpc -I .\eigen SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0i.exe
icpc -I .\eigen -O1 SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0iO1.exe
icpc -I .\eigen -O2 SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0iO2.exe
icpc -I .\eigen -O3 SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0iO3.exe
icpc -I .\eigen -xO SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0iOx.exe
icpc -I .\eigen -fast SC_LOQDhpc1.0.cpp iohpc.cpp def.cpp -o SC_LOQDhpc1.0iOf.exe

  Boundary type options
  1: incoming according to side
  2: incoming according to angle
  3: reflective on all sides
  4: reflective on Left and Bottom, incoming on Right and Top according to side
  5: reflective on Right and Top, incoming on Left and Bottom according to side
  
  BC input order
  according to side : Left, Bottom, Right, Top
  according to angle: quad 1, quad 2, quad 3, quad 4
  
  Quadratures
  S4, S6, S8, S12, S16
  Q20, Q20
  

*/

void Iterations();    // perform source iterations on problem
void angleSweep();          // determine boundary conditions and how to sequence quadrant solutions
void quad1();               // sweep through angles and cells in first quadrant
void quad2();               // sweep through angles and cells in second quadrant
void quad3();               // sweep through angles and cells in third quadrant
void quad4();               // sweep through angles and cells in fourth quadrant
void cellSolution(double, double, double, double, double, double, double, double, double, double&, double&, double& );
void LOQDsolution();          // LOQD solution function
void output(char[]);          // output code
// data output functions

const double pi=3.141592654;
int nx, ny, sn, n_iterations;
int maxiter_lo=1000;
int o_angular=0; // option variables
double epsilon_si=1e-5, epsilon_lo=1e-10; // default tolerances
int N=8;                     // quadrature
double *mu, *eta, *xi, *w;   // quadrature
double * x, * y, * hx, * hy; // grid arrays
double **sigmaT, **sigmaS, **sigmaF, **nuF, **s_ext; // material arrays
int **material;
double **psi , **psi_x , **psi_y; // angular flux
double **phi , **phi_x , **phi_y , **j_x , **j_y; // scalar flux and current solution
double **phiT, **phi_xT, **phi_yT, **j_xT, **j_yT; // scalar flux and current from transport
double **phiL, **phi_xL, **phi_yL; // scalar flux from last iteration
int    kbc;
double bcL, bcR, bcB, bcT;
double **E_xx  , **E_yy  , **E_xy;
double **E_xx_x, **E_yy_x, **E_xy_x;
double **E_xx_y, **E_yy_y, **E_xy_y;
double ***psiL, ***psiR, ***psiB, ***psiT;
double *cL, *cR, *cB, *cT;
double *phiOutL, *phiOutR, *phiOutB, *phiOutT;
double *phiInL, *phiInR, *phiInB, *phiInT;
double *jInL, *jInR, *jInB, *jInT;
// residual data
int ii=0, i_mbal, i_mx, i_my, i_mb, i_mt;
int jj=0, j_mbal, j_mx, j_my, j_ml, j_mr;
double max_res, res_mbal, res_mx, res_my, res_ml, res_mr, res_mb, res_mt;
double res_bal, res_l, res_r, res_b, res_t, res_lbc, res_rbc, res_bbc, res_tbc;
int i_bal, i_l, i_r, i_b, i_t, i_bbc, i_tbc;
int j_bal, j_l, j_r, j_b, j_t, j_lbc, j_rbc;
// iteration data
vector<double> rho_si (1,0.5), err_lo, dt_ho, dt_lo, dt_pc;
vector<int> num_lo;
ofstream outfile;
ofstream datfile;
ofstream temfile;
clock_t t;
clock_t t_lo;
clock_t t_ho;
clock_t t_pc;


//======================================================================================//
//++ Read Input ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

//======================================================================================//
void input(char fn[]) //
{
	char inpf[25]={""}, outf[25]={""};
	const int nl=250;
	char namel[nl], numl[nl], matl[nl];
	int *regnum, *matnum;
	double *xn, *yn, *xp, *yp, *st, *ss, *sf, *nf, *se, *xg, *yg;
	int i, j, k, p, nmat, nreg, ngx=0, ngy=0,  *nxt, *nyt;
	
	// input file name
	strcat (inpf,fn); strcat (inpf,".inp"); // input file name
	cout << inpf << endl;
	
	// open file +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	ifstream infile;
	infile.open (inpf); // open input file
	
	// read in data
	infile.getline(namel, nl); // read in name of the input
	
	// read x grid data
	infile.getline(numl, nl); // x grid edges
	for (i=0; i<nl; i++) {
		if ( isdigit(numl[i]) ) {
			if ( ngx==0 ) {
				ngx=iparse(numl,i,nl); // find number of mesh regions
				nxt=new int[ngx];
				xg=new double[ngx+1];
				p=0;
			}
			else {
				xg[p]=dparse(numl,i,nl); p++;
			}
		}
		if ( numl[i]==';' ) break;
	}
	if ( p!=ngx+1 )	cout<<"xgrid input error\n";
	
	infile.getline(numl, nl); // # cells in x grid
	p=0;
	for (i=0; i<nl; i++) {
		if ( isdigit(numl[i]) ) {
			nxt[p]=iparse(numl,i,nl); p++;
		}
		if ( numl[i]==';' )	break;
	}
	if ( p!=ngx ) cout<<"xgrid input error\n";
	
	// read y grid data
	infile.getline(numl, nl); // read in grid data
	for (i=0; i<nl; i++) {
		if ( isdigit(numl[i]) ) {
			if ( ngy==0 ) {
				ngy=iparse(numl,i,nl);
				nyt=new int[ngy];
				yg=new double[ngy+1];
				p=0;
			}
			else {
				yg[p]=dparse(numl,i,nl); p++; // y grid zones
			}
		}
		if ( numl[i]==';' )	break;
	}
	if ( p!=ngy+1 )	cout<<"ygrid input error\n";
	
	infile.getline(numl, nl); // read in grid data
	p=0;
	for (i=0; i<nl; i++) {
		if ( isdigit(numl[i]) ) {
			nyt[p]=iparse(numl,i,nl); p++; // # cells in y grid zone
		}
		if ( numl[i]==';' ) break;
	}
	if ( p!=ngy ) cout<<"y grid input error\n";
	
	// read in boundary conditions
	p=0;
	infile.getline(matl, nl);
	for (i=0; i<nl; i++) {
		if ( isdigit(matl[i]) ) {
			if ( p==0 )      kbc=iparse(matl,i,nl); // kind of BC
			else if ( p==1 ) bcL=dparse(matl,i,nl); // Left   BC or Quadrant 1
			else if ( p==2 ) bcB=dparse(matl,i,nl); // Bottom BC or Quadrant 2
			else if ( p==3 ) bcR=dparse(matl,i,nl); // Right  BC or Quadrant 3
			else             bcT=dparse(matl,i,nl); // Top    BC or Quadrant 4
			p++;
		}
		if ( matl[i]==';' ) break;
	}
	
	
	infile.getline(numl, nl); // space
	
	infile.getline(numl, nl);  // number of materials
	for (i=0; i<nl; i++) {
		if ( isdigit(numl[i]) ) {
			nmat=iparse(numl,i,nl); break;
		}
	}
	
	// cross sections
	matnum=new int[nmat];
	st=new double[nmat];
	ss=new double[nmat];
	sf=new double[nmat];
	nf=new double[nmat];
	se=new double[nmat];
	
	for (k=0; k<nmat; k++) {
		p=0;
		infile.getline(matl, nl);
		for (i=0; i<nl; i++) {
			if ( isdigit(matl[i]) ) {
				if ( p==0 )  matnum[k]=iparse(matl,i,nl); // Material #
				else if ( p==1 ) st[k]=dparse(matl,i,nl); // sigmaT total cross section
				else if ( p==2 ) ss[k]=dparse(matl,i,nl); // sigmaS scattering cross section
				else if ( p==3 ) sf[k]=dparse(matl,i,nl); // sigmaF fission cross section
				else if ( p==4 ) nf[k]=dparse(matl,i,nl); // nuF
				else if ( p==5 ) se[k]=4.0*pi*dparse(matl,i,nl); // external source
				else cout<<" Material Input Error \n";
				p++;
			}
			if ( matl[i]==';' ) break;
		}
	}
	
	infile.getline(numl, nl);  // number of material regions
	for (i=0; i<nl; i++) {
		if ( isdigit(numl[i]) ) {
			nreg=iparse(numl,i,nl); break;
		}
	}
	
	// material edges
	regnum=new int[nreg];
	xn=new double[nreg];  xp=new double[nreg];
	yn=new double[nreg];  yp=new double[nreg];
	
	for (k=0; k<nreg; k++) {
		p=0;
		infile.getline(matl, nl);
		for (i=0; i<nl; i++) {
			if ( isdigit(matl[i]) ) {
				if ( p==0 )  regnum[k]=iparse(matl,i,nl); // Material Number in Region
				else if ( p==1 ) xn[k]=dparse(matl,i,nl); // Left   Boundary of Region
				else if ( p==2 ) yn[k]=dparse(matl,i,nl); // Bottom Boundary of Region
				else if ( p==3 ) xp[k]=dparse(matl,i,nl); // Right  Boundary of Region
				else if ( p==4 ) yp[k]=dparse(matl,i,nl); // Top    Boundary of Region
				else cout<<" Region Input Error \n";
				p++;
			}
			if ( matl[i]==';' ) break;
		}
	}
	p=0;
	infile.getline(numl, nl); // Read additional data
	for (i=0; i<nl; i++) {
		if ( isdigit(numl[i]) ) {
			if ( p==0 )      epsilon_si=dparse(numl,i,nl); // get convergence criteria
			else if ( p==1 ) epsilon_lo=dparse(numl,i,nl); // get low-order convergence criteria 
			p++;
		}
		if ( matl[i]==';' ) break;
	}
	
	if ( epsilon_si<1e-12 ) {
		cout<<">> Error in SI convergence criteria input!\n"; epsilon_si=1e-5;
	}
	if ( epsilon_lo<1e-14 ) {
		cout<<">> Error in low-order convergence criteria input!\n"; epsilon_lo=1e-10;
	}
	infile.getline(numl, nl); // Read additional data
	for (i=0; i<nl; i++) {
		if ( isdigit(numl[i]) ) {
			N=iparse(numl,i,nl); break;
		}
		if ( matl[i]==';' ) break;
	}
	if ( N!=4 and N!=6 and N!=8 and N!=12 and N!=16 and N!=20 and N!=36 ) {
		cout<<">> Error in Quadrature input!\n"; N=36;
	}
	
	infile.close(); // close input file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	
	// total grid
	nx=0; for (i=0; i<ngx; i++) nx+=nxt[i]; // find total number of cells in x grid
	ny=0; for (i=0; i<ngy; i++) ny+=nyt[i]; // find total number of cells in y grid
	
	x=new (nothrow) double[nx+1]; hx=new (nothrow) double[nx];
	y=new (nothrow) double[ny+1]; hy=new (nothrow) double[ny];
	
	x[0]=xg[0]; p=0;
	for (i=0; i<ngx; i++) {
		for (j=p; j<p+nxt[i]; j++) {
			hx[j]=(xg[i+1]-xg[i])/nxt[i];
			x[j+1]=x[j]+hx[j];
		}
		p+=nxt[i];
	}
	y[0]=yg[0]; p=0;
	for (i=0; i<ngy; i++) {
		for (j=p; j<p+nyt[i]; j++) {
		hy[j]=(yg[i+1]-yg[i])/nyt[i];
		y[j+1]=y[j]+hy[j];
		}
		p+=nyt[i];
	}
	
	// cross section data
	sigmaT=new double*[nx];
	sigmaS=new double*[nx];
	sigmaF=new double*[nx];
	nuF   =new double*[nx];
	s_ext =new double*[nx];
	material =new int*[nx];
	for (i=0; i<nx; i++) {
		sigmaT[i]=new double[ny];
		sigmaS[i]=new double[ny];
		sigmaF[i]=new double[ny];
		nuF[i]   =new double[ny];
		s_ext[i] =new double[ny];
		material[i] =new int[ny];
		for (j=0; j<ny; j++) {
			k=matRegion(matnum, regnum, i, j, nmat, nreg, xn, yn, xp, yp);
			sigmaT[i][j]=st[k];
			sigmaS[i][j]=ss[k];
			sigmaF[i][j]=sf[k];
			nuF[i][j]   =nf[k];
			s_ext[i][j] =se[k];
			material[i][j]=matnum[k];
			if ( sigmaT[i][j]==0.0 ) sigmaT[i][j]=1e-23;
		}
	}
	
	strcat (outf,fn);
	strcat (outf,".out");
	cout<<outf<<endl;
	outfile.open(outf); // open output file. closed in output function
	
	outfile<<"Output of File : "<<outf<<endl;
	outfile<<"2D Transport by Step Characteristics with LOQD\n";
	outfile<<"Version: 2.1\n";
	outfile<<"Programer : Luke Cornejo\n";
	outfile<<"Case Name : "<<namel<<endl;
	// current date/time based on current system
	time_t now = time(0);
	// convert now to string form
	char* dt = ctime(&now);
	
	outfile<<"Program Ran On: "<<dt<<endl;
	
	outfile<<"+-----------------------------------------------+\n";
	outfile<<"Iteration Convergence Criteria: "<<epsilon_si<<endl;
	outfile<<"Low-order Tolerance: "<<epsilon_lo<<endl;
	outfile<<"+-----------------------------------------------+\n";
	
	outfile<<"\n -- Material Properties --\n";
	outfile<<" Material |  Total   |Scattering|  Fission |   nuF    | External \n";
	outfile<<"    #     |   XS     |    XS    |    XS    |          |  Source  \n";
	outfile.precision(6);
	for (k=0; k<nmat; k++) 
		outfile<<setw(10)<<matnum[k]<<"|"<<setw(10)<<st[k]<<"|"<<setw(10)<<ss[k]<<"|"<<setw(10)<<sf[k]<<"|"<<setw(10)<<nf[k]<<"|"<<setw(10)<<se[k]<<endl;
	
	outfile<<"\n -- Material Map -- \n";
	
	for (j=ny-1; j>=0; j--) {
		for (i=0; i<nx; i++) outfile<<setw(3)<<material[i][j];
		outfile<<endl;
	}
	
	
}
//======================================================================================//

//======================================================================================//
//++ Main program ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
int main (int argc, char* argv[])
{
	char fn[50], outf[50]={""};
	int i, j, m;
	double nu1, c;
	strcpy (fn,argv[1]);
	if ( argc!=2 ) cout << "usage: " << argv[0] << " input file name";
	else cout << fn << endl;
	
	input(fn); // get input data
	
	quadSet(); // find quadrature 
	
	initialize(); // initialize memory space for solution
	
	strcat (outf,fn);
	strcat (outf,".temp.csv");
	cout<<outf<<endl;
	temfile.open(outf); // open temporary file
	
	t = clock();    // start timer
	// ------------------------------------------
	Iterations(); // Call Iterations
	// ------------------------------------------
	t = clock() -t; // stop timer
	
	temfile.close(); // close temporary file
	
	output(fn);     // write out solutions
	
	return 0;
}
//======================================================================================//

//======================================================================================//
//++ Iterate to converge on solution +++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void Iterations()
{
	int i, j, ii, jj, m, s;
	double norm_si, norm_siL, phi_sum, res, psi_const=1.0;
	
	
	// Find Initial Factors from constant angular flux
	for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
			E_xx[i][j]=0.0; E_yy[i][j]=0.0; E_xy[i][j]=0.0; phi[i][j]=0.0;
			for (m=0; m<sn; m++) {
				E_xx[i][j]+=mu[m] *mu[m] *psi_const*w[m]; E_yy[i][j]+=eta[m]*eta[m]*psi_const*w[m];
				E_xy[i][j]+=mu[m] *eta[m]*psi_const*w[m]; phi[i][j]+=              psi_const*w[m];
			}
			E_xx[i][j]*=4.0; E_yy[i][j]*=4.0; phi[i][j]*=4.0;
			for (m=0; m<sn; m++) E_xy[i][j]+=-mu[m] *eta[m]*psi_const*w[m];
			E_xy[i][j]*=2.0;
			E_xx[i][j]/=phi[i][j]; E_yy[i][j]/=phi[i][j]; E_xy[i][j]/=phi[i][j];
		}
	}
	for (i=0; i<nx+1; i++) {
		for (j=0; j<ny; j++) {
			E_xx_x[i][j]=0.0; E_yy_x[i][j]=0.0; E_xy_x[i][j]=0.0; phi_x[i][j]=0.0;
			for (m=0; m<sn; m++) {
				E_xx_x[i][j]+=mu[m] *mu[m] *psi_const*w[m]; E_yy_x[i][j]+=eta[m]*eta[m]*psi_const*w[m];
				E_xy_x[i][j]+=mu[m] *eta[m]*psi_const*w[m]; phi_x[i][j]+=              psi_const*w[m];
			}
			E_xx_x[i][j]*=4.0; E_yy_x[i][j]*=4.0; phi_x[i][j]*=4.0;
			for (m=0; m<sn; m++) E_xy_x[i][j]+=-mu[m] *eta[m]*psi_const*w[m];
			E_xy_x[i][j]*=2.0;
			E_xx_x[i][j]/=phi_x[i][j]; E_yy_x[i][j]/=phi_x[i][j]; E_xy_x[i][j]/=phi_x[i][j];
		}
	}
	for (i=0; i<nx; i++) {
		for (j=0; j<ny+1; j++) {
			E_xx_y[i][j]=0.0; E_yy_y[i][j]=0.0; E_xy_y[i][j]=0.0; phi_y[i][j]=0.0;
			for (m=0; m<sn; m++) {
				E_xx_y[i][j]+=mu[m] *mu[m] *psi_const*w[m]; E_yy_y[i][j]+=eta[m]*eta[m]*psi_const*w[m];
				E_xy_y[i][j]+=mu[m] *eta[m]*psi_const*w[m]; phi_y[i][j]+=              psi_const*w[m];
			}
			E_xx_y[i][j]*=4.0; E_yy_y[i][j]*=4.0; phi_y[i][j]*=4.0;
			for (m=0; m<sn; m++) E_xy_y[i][j]+=-mu[m] *eta[m]*psi_const*w[m];
			E_xy_y[i][j]*=2.0;
			E_xx_y[i][j]/=phi_y[i][j]; E_yy_y[i][j]/=phi_y[i][j]; E_xy_y[i][j]/=phi_y[i][j];
		}
	}
	// find boundary conditions values
	for (i=0; i<nx; i++) {
		cB[i]=0.0; phiInB[i]=0.0; phiOutB[i]=0.0; jInB[i]=0.0;
		cT[i]=0.0; phiInT[i]=0.0; phiOutT[i]=0.0; jInT[i]=0.0;
		for (m=0; m<sn; m++) {
			phiInB[i]+= (psi_const+psi_const)*w[m]; phiOutB[i]+= (psi_const+psi_const)*w[m];
			phiInT[i]+= (psi_const+psi_const)*w[m]; phiOutT[i]+= (psi_const+psi_const)*w[m];
			jInB[i]  += (psi_const+psi_const)*w[m]*eta[m]; cB[i]+=-(psi_const+psi_const)*w[m]*eta[m];
			jInT[i]  +=-(psi_const+psi_const)*w[m]*eta[m]; cT[i]+= (psi_const+psi_const)*w[m]*eta[m];
			psiB[i][m][3]=psi_const; psiB[i][m][4]=psi_const;
			psiT[i][m][1]=psi_const; psiT[i][m][2]=psi_const;
		}
		cB[i]/=phiOutB[i]; cT[i]/=phiOutT[i];
	}
	for (j=0; j<ny; j++) {
		cL[j]=0.0; phiInL[j]=0.0; phiOutL[j]=0.0; jInL[j]=0.0;
		cR[j]=0.0; phiInR[j]=0.0; phiOutR[j]=0.0; jInR[j]=0.0;
		for (m=0; m<sn; m++) {
			phiInL[j]+= (psi_const+psi_const)*w[m]; phiOutL[j]+= (psi_const+psi_const)*w[m];
			phiInR[j]+= (psi_const+psi_const)*w[m]; phiOutR[j]+= (psi_const+psi_const)*w[m];
			jInL[j]  += (psi_const+psi_const)*w[m]*mu[m]; cL[j]+=-(psi_const+psi_const)*w[m]*mu[m];
			jInR[j]  +=-(psi_const+psi_const)*w[m]*mu[m]; cR[j]+= (psi_const+psi_const)*w[m]*mu[m];
			psiL[j][m][2]=psi_const; psiL[j][m][3]=psi_const;
			psiR[j][m][1]=psi_const; psiR[j][m][4]=psi_const;
		}
		cL[j]/=phiOutL[j]; cR[j]/=phiOutR[j];
	}
	// Find Low-order solution
	
	LOQDsolution(); // call LOQD function
	
	
	// Find new norm
	norm_si=0;
	for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
			if ( phi[i][j]>norm_si ) norm_si=phi[i][j];
		}
	}
	
	norm_siL=norm_si;
	s=0;
	// begin iterations
	while ( norm_si>epsilon_si*(1/rho_si[s]-1) ) { //========================================================================
		s++;
		cout<<"Iteration # "<<s<<endl;
		// set previous iteration to phiLast
		for (i=0; i<nx; i++) {
			for (j=0; j<ny; j++) {
				phiL[i][j]=phi[i][j];
			}
		}
		for (i=0; i<nx+1; i++) {
			for (j=0; j<ny; j++) {
				phi_xL[i][j]=phi_x[i][j];
			}
		}
		for (i=0; i<nx; i++) {
			for (j=0; j<ny+1; j++) {
				phi_yL[i][j]=phi_y[i][j];
			}
		}
	
		norm_siL=norm_si; // set previous norm to normLast
		
		temfile<<"Iteration # "<<s<<endl;
		t_ho=clock(); // start high-order timer
		angleSweep(); // perform sweep through angles >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		t_ho=clock()-t_ho; // stop high-order timer
		dt_ho.push_back(((double)t_ho)/CLOCKS_PER_SEC); // add high-order solution time to vector
		

		// Calculate Eddington Factors
		for (i=0; i<nx; i++) {
			for (j=0; j<ny; j++) {
				E_xx[i][j]/=phiT[i][j]; E_yy[i][j]/=phiT[i][j]; E_xy[i][j]/=phiT[i][j];
			}
		}
		for (i=0; i<nx+1; i++) {
			for (j=0; j<ny; j++) {
				E_xx_x[i][j]/=phi_xT[i][j]; E_yy_x[i][j]/=phi_xT[i][j]; E_xy_x[i][j]/=phi_xT[i][j];
			}
		}
		for (i=0; i<nx; i++) {
			for (j=0; j<ny+1; j++) {
				E_xx_y[i][j]/=phi_yT[i][j]; E_yy_y[i][j]/=phi_yT[i][j]; E_xy_y[i][j]/=phi_yT[i][j];
			}
		}
		// find boundary conditions values
		for (i=0; i<nx; i++) {
			cB[i]=0.0; phiInB[i]=0.0; phiOutB[i]=0.0; jInB[i]=0.0;
			cT[i]=0.0; phiInT[i]=0.0; phiOutT[i]=0.0; jInT[i]=0.0;
			for (m=0; m<sn; m++) {
				phiInB[i] += (psiB[i][m][1]+psiB[i][m][2])*w[m];
				phiInT[i] += (psiT[i][m][3]+psiT[i][m][4])*w[m];
				jInB[i]   += (psiB[i][m][1]+psiB[i][m][2])*w[m]*eta[m];
				jInT[i]   +=-(psiT[i][m][3]+psiT[i][m][4])*w[m]*eta[m];
				phiOutB[i]+= (psiB[i][m][3]+psiB[i][m][4])*w[m];
				phiOutT[i]+= (psiT[i][m][1]+psiT[i][m][2])*w[m];
				cB[i]     +=-(psiB[i][m][3]+psiB[i][m][4])*w[m]*eta[m];
				cT[i]     += (psiT[i][m][1]+psiT[i][m][2])*w[m]*eta[m];
			}
			if ( phiOutB[i]==0.0 ) cB[i]=0;
			else cB[i]/=phiOutB[i];
			if ( phiOutT[i]==0.0 ) cT[i]=0;
			else cT[i]/=phiOutT[i];
		}
		for (j=0; j<ny; j++) {
			cL[j]=0.0; phiInL[j]=0.0; phiOutL[j]=0.0; jInL[j]=0.0;
			cR[j]=0.0; phiInR[j]=0.0; phiOutR[j]=0.0; jInR[j]=0.0;
			for (m=0; m<sn; m++) {
				phiInL[j] += (psiL[j][m][1]+psiL[j][m][4])*w[m];
				phiInR[j] += (psiR[j][m][2]+psiR[j][m][3])*w[m];
				jInL[j]   += (psiL[j][m][1]+psiL[j][m][4])*w[m]*mu[m];
				jInR[j]   +=-(psiR[j][m][2]+psiR[j][m][3])*w[m]*mu[m];
				phiOutL[j]+= (psiL[j][m][2]+psiL[j][m][3])*w[m];
				phiOutR[j]+= (psiR[j][m][1]+psiR[j][m][4])*w[m];
				cL[j]     +=-(psiL[j][m][2]+psiL[j][m][3])*w[m]*mu[m];
				cR[j]     += (psiR[j][m][1]+psiR[j][m][4])*w[m]*mu[m];
			}
			if ( phiOutL[j]==0.0 ) cL[j]=0;
			else cL[j]/=phiOutL[j];
			if ( phiOutR[j]==0.0 ) cR[j]=0;
			else cR[j]/=phiOutR[j];
		}
		
		// In case of reflective BC set edge values to zero
		switch (kbc) {
			case 3: // BC type 3 Reflective on all sides 
				for (i=0; i<nx; i++) {
					cB[i]=0; phiInB[i]=0; jInB[i]=0;
					cT[i]=0; phiInT[i]=0; jInT[i]=0;
				}
				for (j=0; j<ny; j++) {
					cL[j]=0; phiInL[j]=0; jInL[j]=0;
					cR[j]=0; phiInR[j]=0; jInR[j]=0;
				}
				break;
			case 4: // BC type 4 Reflective on Bottom and Left
				for (i=0; i<nx; i++) {
					cB[i]=0; phiInB[i]=0; jInB[i]=0;
				}
				for (j=0; j<ny; j++) {
					cL[j]=0; phiInL[j]=0; jInL[j]=0;
				}
				break;
			case 5: // BC type 5 Reflective on Top and Right
				for (i=0; i<nx; i++) {
					cT[i]=0; phiInT[i]=0; jInT[i]=0;
				}
				for (j=0; j<ny; j++) {
					cR[j]=0; phiInR[j]=0; jInR[j]=0;
				}
				break;
			default:
				break;
		}
		
		// Calculate Transport Residuals
		max_res=0;
		for (i=0; i<nx; i++) {
			for (j=0; j<ny; j++) {
				res=abs((j_xT[i+1][j]-j_xT[i][j])*hy[j]+(j_yT[i][j+1]-j_yT[i][j])*hx[i]+
					(sigmaT[i][j]*phiT[i][j]-(sigmaS[i][j]+nuF[i][j]*sigmaF[i][j])*phiL[i][j]-s_ext[i][j])*hx[i]*hy[j]);
				if ( res>max_res ) {
					max_res=res; ii=i; jj=j;
				}
			}
		}
		t_lo=clock(); // start low-order timer
		LOQDsolution(); // call LOQD function
		t_lo=clock()-t_lo; // stop low-order timer
		dt_lo.push_back(((double)t_lo)/CLOCKS_PER_SEC); // add high-order solution time to vector
		cout<<"Iteration Completed in "<<dt_ho[s-1]+dt_lo[s-1]<<" sec\n";
		
		// Find new norm
		norm_si=0;
		for (i=0; i<nx; i++) {
			for (j=0; j<ny; j++) {
				if ( abs(phi[i][j]-phiL[i][j])>norm_si ) norm_si=abs(phi[i][j]-phiL[i][j]);
			}
		}
		rho_si.push_back(norm_si/norm_siL);
		
	}
	n_iterations=s;
	
	// for periodic case normalize solution
	if ( kbc==3 ) {
		phi_sum=0.0;
		for (i=0; i<nx; i++) {
			for (j=0; j<ny; j++) phi_sum+=phi[i][j]*hx[i]*hy[j];
		}
		phi_sum/=4*pi; // normalize to integrate to 4pi
		for (i=0; i<nx; i++) {
			for (j=0; j<ny; j++) {
				phi[i][j] /=phi_sum;
				phiT[i][j]/=phi_sum;
			}
		}
		for (i=0; i<nx+1; i++) {
			for (j=0; j<ny; j++) {
				phi_x[i][j] /=phi_sum;
				j_x[i][j]   /=phi_sum;
				phi_xT[i][j]/=phi_sum;
				j_xT[i][j]  /=phi_sum;
			}
		}
		for (i=0; i<nx; i++) {
			for (j=0; j<ny+1; j++) {
				phi_y[i][j] /=phi_sum;
				j_y[i][j]   /=phi_sum;
				phi_yT[i][j]/=phi_sum;
				j_yT[i][j]  /=phi_sum;
			}
		}
	}
}
//======================================================================================//

//======================================================================================//
//++ Determine how to sweep thought cells ++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void angleSweep()
{
	int i, j, m;
	cout<<"Transport Sweep Started : ";
	// Zero out Eddington Factors and Currents
	for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
			phiT[i][j]=0.0;
			E_xx[i][j]=0.0; E_yy[i][j]=0.0; E_xy[i][j]=0.0;
		}
	}
	for (i=0; i<nx+1; i++) {
		for (j=0; j<ny; j++) {
			phi_xT[i][j]=0.0; j_xT[i][j]=0.0;
			E_xx_x[i][j]=0.0; E_yy_x[i][j]=0.0; E_xy_x[i][j]=0.0;
		}
	}
	for (i=0; i<nx; i++) {
		for (j=0; j<ny+1; j++) {
			phi_yT[i][j]=0.0; j_yT[i][j]=0.0;
			E_xx_y[i][j]=0.0; E_yy_y[i][j]=0.0; E_xy_y[i][j]=0.0;
		}
	}
	
	switch ( kbc ) {
	case 1: // incoming boundary conditions on face only 111111111111111111111111111111111111111111111111111111111111111111
		for (m=0; m<sn; m++) {
			for (i=0; i<nx; i++) { // top and bottom boundary conditions
				psiB[i][m][1]=bcB; // Quad 1 Bottom boundary condition
				psiB[i][m][2]=bcB; // Quad 2 Bottom boundary condition
				psiT[i][m][3]=bcT; // Quad 3 Top boundary condition
				psiT[i][m][4]=bcT; // Quad 4 Top boundary condition
			}
			for (j=0; j<ny; j++) { // left and right boundary conditions
				psiL[j][m][1]=bcL; // Quad 1 Left boundary condition
				psiL[j][m][4]=bcL; // Quad 4 Left boundary condition
				psiR[j][m][2]=bcR; // Quad 2 Right boundary condition
				psiR[j][m][3]=bcR; // Quad 3 Right boundary condition
			}
		}
		
		// Start solution sweep///////////////////////////////////////
		quad1(); // solution sweep through quadrant 1
		quad2(); // solution sweep through quadrant 2
		quad3(); // solution sweep through quadrant 3
		quad4(); // solution sweep through quadrant 4
		break;
	case 2: // incoming boundary conditions in Quadrants 22222222222222222222222222222222222222222222222222222222222222222222
		for (m=0; m<sn; m++) {
			for (i=0; i<nx; i++) { // top and bottom boundary conditions
				psiB[i][m][1]=bcL; // Quad 1 Bottom boundary condition
				psiB[i][m][2]=bcB; // Quad 2 Bottom boundary condition
				psiT[i][m][3]=bcR; // Quad 3 Top    boundary condition
				psiT[i][m][4]=bcT; // Quad 4 Top    boundary condition
			}
			for (j=0; j<ny; j++) { // left and right boundary conditions
				psiL[j][m][1]=bcL; // Quad 1 Left  boundary condition
				psiL[j][m][4]=bcT; // Quad 4 Left  boundary condition
				psiR[j][m][2]=bcB; // Quad 2 Right boundary condition
				psiR[j][m][3]=bcR; // Quad 3 Right boundary condition
			}
		}
		
		// Start solution sweep///////////////////////////////////////
		quad1(); // solution sweep through quadrant 1
		quad2(); // solution sweep through quadrant 2
		quad3(); // solution sweep through quadrant 3
		quad4(); // solution sweep through quadrant 4
		break;
	case 3: // All Reflective BC 333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

		// Start solution sweep
		for (m=0; m<sn; m++) { // quad 1
			for (i=0; i<nx; i++) {
				psiB[i][m][1]=psiB[i][m][4]; // Quad 1 Bottom Reflective BC
				psiB[i][m][2]=psiB[i][m][3]; // Quad 2 Bottom Reflective BC
				psiT[i][m][3]=psiT[i][m][2]; // Quad 3 Top Reflective BC
				psiT[i][m][4]=psiT[i][m][1]; // Quad 4 Top Reflective BC
			}
			for (j=0; j<ny; j++) {
				psiL[j][m][1]=psiL[j][m][2]; // Quad 1 Left Reflective BC
				psiR[j][m][2]=psiR[j][m][1]; // Quad 2 Right Reflective BC
				psiR[j][m][3]=psiR[j][m][4]; // Quad 3 Right Reflective BC
				psiL[j][m][4]=psiL[j][m][3]; // Quad 4 Left Reflective BC
			}
		}
		
		quad1(); // solution sweep through quadrant 1
		quad2(); // solution sweep through quadrant 2
		quad3(); // solution sweep through quadrant 3
		quad4(); // solution sweep through quadrant 4
			
		break;
	case 4: //Reflective BC on bottom and left, face BC 444444444444444444444444444444444444444444444444444444444444444444444444
		// Start solution sweep ///////////////////////////////////////////////////////////
		for (m=0; m<sn; m++) { // quad 3
			for (i=0; i<nx; i++) psiT[i][m][3]=bcT; // Quad 3 Top BC
			for (j=0; j<ny; j++) psiR[j][m][3]=bcR; // Quad 3 Right BC
		}
		quad3(); // solution sweep through quadrant 3
		
		for (m=0; m<sn; m++) { // quad 4
			for (i=0; i<nx; i++) psiT[i][m][4]=bcT;           // Quad 4 Top BC
			for (j=0; j<ny; j++) psiL[j][m][4]=psiL[j][m][3]; // Quad 4 Left Reflective BC
		}
		quad4(); // solution sweep through quadrant 4
		
		for (m=0; m<sn; m++) { // quad 2
			for (i=0; i<nx; i++) psiB[i][m][2]=psiB[i][m][3]; // Quad 2 Bottom Reflective BC
			for (j=0; j<ny; j++) psiR[j][m][2]=bcR;           // Quad 2 Right BC
		}
		quad2(); // solution sweep through quadrant 2
		
		for (m=0; m<sn; m++) { // quad 1
			for (i=0; i<nx; i++) psiB[i][m][1]=psiB[i][m][4]; // Quad 1 Bottom Reflective BC
			for (j=0; j<ny; j++) psiL[j][m][1]=psiL[j][m][2]; // Quad 1 Left Reflective BC
		}
		quad1(); // solution sweep through quadrant 1
		break;
	case 5: // Reflective BC on top and right, face BC 55555555555555555555555555555555555555555555555555555555555555555555555555
		// Start solution sweep ////////////////////////////////////////////////////////
		for (m=0; m<sn; m++) {
			for (i=0; i<nx; i++) psiB[i][m][1]=bcB; // Quad 1 Bottom boundary condition
			for (j=0; j<ny; j++) psiL[j][m][1]=bcL; // Quad 1 Left boundary condition
		}
		quad1(); // solution sweep through quadrant 1
		
		for (m=0; m<sn; m++) {
			for (i=0; i<nx; i++) psiB[i][m][2]=bcB;           // Quad 2 Bottom BC
			for (j=0; j<ny; j++) psiR[j][m][2]=psiR[j][m][1]; // Quad 2 Right reflective BC
		}
		quad2(); // solution sweep through quadrant 2
		
		for (m=0; m<sn; m++) { // quad 4
			for (i=0; i<nx; i++) psiT[i][m][4]=psiT[i][m][1]; // Quad 4 Top reflective BC
			for (j=0; j<ny; j++) psiL[j][m][4]=bcL;           // Quad 4 Left BC
		}
		quad4(); // solution sweep through quadrant 4
		
		for (m=0; m<sn; m++) { // quad 3
			for (i=0; i<nx; i++) psiT[i][m][3]=psiT[i][m][2]; // Quad 3 Top reflective BC
			for (j=0; j<ny; j++) psiR[j][m][3]=psiR[j][m][4]; // Quad 3 Right reflective BC
		}
		quad3(); // solution sweep through quadrant 3
		break;
	default:
		cout<<"bad boundary conditions: incorrect boundary type"<<endl;
		break;
	
	}
	cout<<"Completed \n";
}
//======================================================================================//

//======================================================================================//
//++ sweep through angles and cells in each angular quadrant +++++++++++++++++++++++++++//
//======================================================================================//
void quad1() // solution in quadrant 1
{
	int i, j, m, outw=16;
	double psiA, SA;
	double omega_x, omega_y;
	
	for (m=0; m<sn; m++) { // first quadrant
		omega_x=mu[m]; omega_y=eta[m];
		
		for (i=0; i<nx; i++) psi_y[i][0]=psiB[i][m][1]; // Bottom In BC
		for (j=0; j<ny; j++) psi_x[0][j]=psiL[j][m][1]; // Left In BC
		
		for (j=0; j<ny; j++) { // bottom to top
			for (i=0; i<nx; i++) { // left to right
				//psiInL=psi_x[i][j]; // incoming angular flux on the left
				//psiInB=psi_y[i][j]; // incoming angular flux on the bottom
				SA=((sigmaS[i][j] + nuF[i][j]*sigmaF[i][j])*phiL[i][j]+s_ext[i][j])/(4*pi); // source in the cell
				cellSolution( psi_y[i][j], psi_x[i][j], SA, sigmaT[i][j], mu[m], eta[m], xi[m], hx[i], hy[j], psi_y[i][j+1], psi_x[i+1][j], psiA );
				//psi_x[i+1][j]=psiOutR; // outgoing angular flux on the right
				//psi_y[i][j+1]=psiOutT; // outgoing angular flux on the top
				psi[i][j]=psiA;        // cell average angular flux
				// Calculate cell center values
				phiT[i][j]+=psiA*w[m];
				E_xx[i][j]+=omega_x*omega_x*psiA*w[m];
				E_yy[i][j]+=omega_y*omega_y*psiA*w[m];
				E_xy[i][j]+=omega_x*omega_y*psiA*w[m];
			}
		}
		// Calculate Cell Edge values
		for (i=0; i<nx+1; i++) {
			for (j=0; j<ny; j++) {
				phi_xT[i][j]+=psi_x[i][j]*w[m];
				j_xT[i][j]  +=omega_x*psi_x[i][j]*w[m];
				E_xx_x[i][j]+=omega_x*omega_x*psi_x[i][j]*w[m];
				E_yy_x[i][j]+=omega_y*omega_y*psi_x[i][j]*w[m];
				E_xy_x[i][j]+=omega_x*omega_y*psi_x[i][j]*w[m];
			}
		}
		for (i=0; i<nx; i++) {
			for (j=0; j<ny+1; j++) {
				phi_yT[i][j]+=psi_y[i][j]*w[m];
				j_yT[i][j]  +=omega_y*psi_y[i][j]*w[m];
				E_xx_y[i][j]+=omega_x*omega_x*psi_y[i][j]*w[m];
				E_yy_y[i][j]+=omega_y*omega_y*psi_y[i][j]*w[m];
				E_xy_y[i][j]+=omega_x*omega_y*psi_y[i][j]*w[m];
			}
		}
		
		for (i=0; i<nx; i++) psiT[i][m][1]=psi_y[i][ny]; // Top Out BC
		for (j=0; j<ny; j++) psiR[j][m][1]=psi_x[nx][j]; // Right Out BC
		
		// option to print out angular flux
		switch ( o_angular ) {
		case 1:
			temfile<<" Quadrant 1 Direction # ,"<<m<<",\n";
			temfile<<"  Qmega_x ,"<<print_csv(omega_x)<<"  Omega_y ,"<<print_csv(omega_y)<<"  Weight ,"<<print_csv(w[m])<<endl;
			temfile<<" -- Cell Averaged Angular Flux -- \n";
			write_cell_average_dat(psi, outw, temfile); // call function to write out cell average scalar flux
			temfile<<" -- X Vertical Cell Edge Angular Flux -- \n";
			write_cell_edge_x_dat(psi_x, outw, temfile); // call function to write out cell edge scalar flux on x grid
			temfile<<" -- Y Horizontal Cell Edge Angular Flux -- \n";
			write_cell_edge_y_dat(psi_y, outw, temfile); // call function to write out cell edge scalar flux on y grid
			break;
		case 0:
			
		default:
			break;
		}
	}
}
//======================================================================================//
void quad2() // solution in quadrant 2
{
	int i, j, m, outw=16;
	double psiA, SA;
	double omega_x, omega_y;
	
	for (m=0; m<sn; m++) { // second quadrant
		omega_x=-mu[m]; omega_y=eta[m];
		
		for (i=0; i<nx; i++) psi_y[i][0]=psiB[i][m][2]; // Bottom In BC
		for (j=0; j<ny; j++) psi_x[nx][j]=psiR[j][m][2]; // Right In BC
		
		for (j=0; j<ny; j++) { // bottom to top
			for (i=nx-1; i>=0; i--) { // right to left
				//psiInL=psi_x[i+1][j]; // incoming angular flux on the left
				//psiInB=psi_y[i][j];   // incoming angular flux on the bottom
				SA=((sigmaS[i][j] + nuF[i][j]*sigmaF[i][j])*phiL[i][j]+s_ext[i][j])/(4*pi); // source in the cell
				cellSolution( psi_y[i][j], psi_x[i+1][j], SA, sigmaT[i][j], mu[m], eta[m], xi[m], hx[i], hy[j], psi_y[i][j+1], psi_x[i][j], psiA );
				//psi_x[i][j]=psiOutR;   // outgoing angular flux on the right
				//psi_y[i][j+1]=psiOutT; // outgoing angular flux on the top
				psi[i][j]=psiA;        // cell average angular flux
				// Calculate cell center values
				phiT[i][j]+=psiA*w[m];
				E_xx[i][j]+=omega_x*omega_x*psiA*w[m];
				E_yy[i][j]+=omega_y*omega_y*psiA*w[m];
				E_xy[i][j]+=omega_x*omega_y*psiA*w[m];
			}
		}
		// Calculate Cell Edge values
		for (i=0; i<nx+1; i++) {
			for (j=0; j<ny; j++) {
				phi_xT[i][j]+=psi_x[i][j]*w[m];
				j_xT[i][j]  +=omega_x*psi_x[i][j]*w[m];
				E_xx_x[i][j]+=omega_x*omega_x*psi_x[i][j]*w[m];
				E_yy_x[i][j]+=omega_y*omega_y*psi_x[i][j]*w[m];
				E_xy_x[i][j]+=omega_x*omega_y*psi_x[i][j]*w[m];
			}
		}
		for (i=0; i<nx; i++) {
			for (j=0; j<ny+1; j++) {
				phi_yT[i][j]+=psi_y[i][j]*w[m];
				j_yT[i][j]  +=omega_y*psi_y[i][j]*w[m];
				E_xx_y[i][j]+=omega_x*omega_x*psi_y[i][j]*w[m];
				E_yy_y[i][j]+=omega_y*omega_y*psi_y[i][j]*w[m];
				E_xy_y[i][j]+=omega_x*omega_y*psi_y[i][j]*w[m];
			}
		}
		
		// Out BC
		for (i=0; i<nx; i++) psiT[i][m][2]=psi_y[i][ny]; // Top Out BC
		for (j=0; j<ny; j++) psiL[j][m][2]=psi_x[0][j]; // Left Out BC
		
		// option to print out angular flux
		switch ( o_angular ) {
		case 1:
			temfile<<" Quadrant 2 Direction # ,"<<m<<",\n";
			temfile<<"  Qmega_x ,"<<print_csv(omega_x)<<"  Omega_y ,"<<print_csv(omega_y)<<"  Weight ,"<<print_csv(w[m])<<endl;
			temfile<<" -- Cell Averaged Angular Flux -- \n";
			write_cell_average_dat(psi, outw, temfile); // call function to write out cell average scalar flux
			temfile<<" -- X Vertical Cell Edge Angular Flux -- \n";
			write_cell_edge_x_dat(psi_x, outw, temfile); // call function to write out cell edge scalar flux on x grid
			temfile<<" -- Y Horizontal Cell Edge Angular Flux -- \n";
			write_cell_edge_y_dat(psi_y, outw, temfile); // call function to write out cell edge scalar flux on y grid
			break;
		case 0:
			
		default:
			break;
		}
		
	}
}
//======================================================================================//
void quad3() // solution in quadrant 3
{
	int i, j, m, outw=16;
	double psiA, SA;
	double omega_x, omega_y;
	
	for (m=0; m<sn; m++) { // third quadrant
		omega_x=-mu[m]; omega_y=-eta[m];
		
		for (i=0; i<nx; i++) psi_y[i][ny]=psiT[i][m][3]; // Top In BC
		for (j=0; j<ny; j++) psi_x[nx][j]=psiR[j][m][3]; // Right In BC
		
		for (j=ny-1; j>=0; j--) { // top to bottom
			for (i=nx-1; i>=0; i--) { // right to left
				//psiInL=psi_x[i+1][j]; // incoming angular flux on the left
				//psiInB=psi_y[i][j+1]; // incoming angular flux on the bottom
				SA=((sigmaS[i][j] + nuF[i][j]*sigmaF[i][j])*phiL[i][j]+s_ext[i][j])/(4*pi); // source in the cell
				cellSolution( psi_y[i][j+1], psi_x[i+1][j], SA, sigmaT[i][j], mu[m], eta[m], xi[m], hx[i], hy[j], psi_y[i][j], psi_x[i][j], psiA );
				//psi_x[i][j]=psiOutR; // outgoing angular flux on the right
				//psi_y[i][j]=psiOutT; // outgoing angular flux on the top
				psi[i][j]=psiA;      // cell average angular flux
				// Calculate cell center values
				phiT[i][j]+=psiA*w[m];
				E_xx[i][j]+=omega_x*omega_x*psiA*w[m];
				E_yy[i][j]+=omega_y*omega_y*psiA*w[m];
				E_xy[i][j]+=omega_x*omega_y*psiA*w[m];
			}
		}
		// Calculate Cell Edge values
		for (i=0; i<nx+1; i++) {
			for (j=0; j<ny; j++) {
				phi_xT[i][j]+=psi_x[i][j]*w[m];
				j_xT[i][j]  +=omega_x*psi_x[i][j]*w[m];
				E_xx_x[i][j]+=omega_x*omega_x*psi_x[i][j]*w[m];
				E_yy_x[i][j]+=omega_y*omega_y*psi_x[i][j]*w[m];
				E_xy_x[i][j]+=omega_x*omega_y*psi_x[i][j]*w[m];
			}
		}
		for (i=0; i<nx; i++) {
			for (j=0; j<ny+1; j++) {
				phi_yT[i][j]+=psi_y[i][j]*w[m];
				j_yT[i][j]  +=omega_y*psi_y[i][j]*w[m];
				E_xx_y[i][j]+=omega_x*omega_x*psi_y[i][j]*w[m];
				E_yy_y[i][j]+=omega_y*omega_y*psi_y[i][j]*w[m];
				E_xy_y[i][j]+=omega_x*omega_y*psi_y[i][j]*w[m];
			}
		}
		
		for (i=0; i<nx; i++) psiB[i][m][3]=psi_y[i][0]; // Bottom Out BC
		for (j=0; j<ny; j++) psiL[j][m][3]=psi_x[0][j]; // Left Out BC
		
		// option to print out angular flux
		switch ( o_angular ) {
		case 1:
			temfile<<" Quadrant 3 Direction # ,"<<m<<",\n";
			temfile<<"  Qmega_x ,"<<print_csv(omega_x)<<"  Omega_y ,"<<print_csv(omega_y)<<"  Weight ,"<<print_csv(w[m])<<endl;
			temfile<<" -- Cell Averaged Angular Flux -- \n";
			write_cell_average_dat(psi, outw, temfile); // call function to write out cell average scalar flux
			temfile<<" -- X Vertical Cell Edge Angular Flux -- \n";
			write_cell_edge_x_dat(psi_x, outw, temfile); // call function to write out cell edge scalar flux on x grid
			temfile<<" -- Y Horizontal Cell Edge Angular Flux -- \n";
			write_cell_edge_y_dat(psi_y, outw, temfile); // call function to write out cell edge scalar flux on y grid
			break;
		case 0:
			
		default:
			break;
		}
	}
}
//======================================================================================//
void quad4() // solution in quadrant 4
{
	int i, j, m, outw=16;
	double psiA, SA;
	double omega_x, omega_y;
	
	for (m=0; m<sn; m++) { // fourth quadrant
		omega_x=mu[m]; omega_y=-eta[m];
		
		for (i=0; i<nx; i++) psi_y[i][ny]=psiT[i][m][4]; // Top In BC
		for (j=0; j<ny; j++) psi_x[0][j]=psiL[j][m][4]; // Left In BC
		
		for (j=ny-1; j>=0; j--) { // top to bottom
			for (i=0; i<nx; i++) { // left to right
				//psiInL=psi_x[i][j];   // incoming angular flux on the left
				//psiInB=psi_y[i][j+1]; // incoming angular flux on the bottom
				SA=((sigmaS[i][j] + nuF[i][j]*sigmaF[i][j])*phiL[i][j]+s_ext[i][j])/(4*pi); // source in the cell
				cellSolution( psi_y[i][j+1], psi_x[i][j], SA, sigmaT[i][j], mu[m], eta[m], xi[m], hx[i], hy[j], psi_y[i][j], psi_x[i+1][j], psiA );
				//psi_x[i+1][j]=psiOutR; // outgoing angular flux on the right
				//psi_y[i][j]=psiOutT;   // outgoing angular flux on the top
				psi[i][j]=psiA;        // cell average angular flux
				// Calculate cell center values
				phiT[i][j]+=psiA*w[m];
				E_xx[i][j]+=omega_x*omega_x*psiA*w[m];
				E_yy[i][j]+=omega_y*omega_y*psiA*w[m];
				E_xy[i][j]+=omega_x*omega_y*psiA*w[m];
			}
		}
		// Calculate Cell Edge values
		for (i=0; i<nx+1; i++) {
			for (j=0; j<ny; j++) {
				phi_xT[i][j]+=psi_x[i][j]*w[m];
				j_xT[i][j]  +=omega_x*psi_x[i][j]*w[m];
				E_xx_x[i][j]+=omega_x*omega_x*psi_x[i][j]*w[m];
				E_yy_x[i][j]+=omega_y*omega_y*psi_x[i][j]*w[m];
				E_xy_x[i][j]+=omega_x*omega_y*psi_x[i][j]*w[m];
			}
		}
		for (i=0; i<nx; i++) {
			for (j=0; j<ny+1; j++) {
				phi_yT[i][j]+=psi_y[i][j]*w[m];
				j_yT[i][j]  +=omega_y*psi_y[i][j]*w[m];
				E_xx_y[i][j]+=omega_x*omega_x*psi_y[i][j]*w[m];
				E_yy_y[i][j]+=omega_y*omega_y*psi_y[i][j]*w[m];
				E_xy_y[i][j]+=omega_x*omega_y*psi_y[i][j]*w[m];
			}
		}
		
		for (i=0; i<nx; i++) psiB[i][m][4]=psi_y[i][0]; // Bottom Out BC
		for (j=0; j<ny; j++) psiR[j][m][4]=psi_x[nx][j]; // Right Out BC
		
		// option to print out angular flux
		switch ( o_angular ) {
		case 1:
			temfile<<" Quadrant 4 Direction # ,"<<m<<",\n";
			temfile<<"  Qmega_x ,"<<print_csv(omega_x)<<"  Omega_y ,"<<print_csv(omega_y)<<"  Weight ,"<<print_csv(w[m])<<endl;
			temfile<<" -- Cell Averaged Angular Flux -- \n";
			write_cell_average_dat(psi, outw, temfile); // call function to write out cell average scalar flux
			temfile<<" -- X Vertical Cell Edge Angular Flux -- \n";
			write_cell_edge_x_dat(psi_x, outw, temfile); // call function to write out cell edge scalar flux on x grid
			temfile<<" -- Y Horizontal Cell Edge Angular Flux -- \n";
			write_cell_edge_y_dat(psi_y, outw, temfile); // call function to write out cell edge scalar flux on y grid
			break;
		case 0:
			
		default:	
			break;
		}
	}
}
//======================================================================================//

//======================================================================================//
//++ function to solve transport in a single general cell ++++++++++++++++++++++++++++++//
//======================================================================================//
void cellSolution(double psiInB, double psiInL, double SA, double sigma, double mut, double etat, double xit, 
double LT, double LR, double& psiOutT, double& psiOutR, double& psiA  )
{
	double epsilon, exp_epsilon, epsilon_2, mup, muc, mu, du;
	double psiOut1, psiOut2, psiOut3, psiA1, psiA2, psiA3;
	double A1, A2, Lout1, Lout2, Lout3;
	
	mup=sqrt(1-xit*xit);             // mu'
	muc=LT/sqrt(LT*LT+LR*LR);        // mu of cell
	mu=mut/sqrt(mut*mut+etat*etat); // projection onto x-y plane
	
	if ( abs(mu-muc)<1e-15 ) { // ray passes through both corners of cell
		du=sqrt(LT*LT+LR*LR);
		if ( sigma<1e-10 ) {
			// triangle A
			psiOutT=psiInL+SA*du/mup/2.0; // find out going angular flux
			psiA1  =psiInL+SA*du/mup/3; // find cell angular flux
			
			// triangle C
			psiOutR=psiInB+SA*du/mup/2.0; // find out going angular flux
			psiA3  =psiInB+SA*du/mup/3; // find cell angular flux
			
			psiA=0.5*(psiInL+psiInB)+SA*du/mup/3.0;
		}
		else {
			epsilon=sigma*du/mup; // optical thickness
			exp_epsilon=exp(-epsilon); // exponent of epsilon
			epsilon_2=epsilon*epsilon; // square of epsilon
			// triangle A
			psiOutT=(psiInL*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1)/sigma)/epsilon; // find out going angular flux
			psiA1=2*(psiInL*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon)/sigma)/epsilon_2; // find cell angular flux
			
			// triangle C
			psiOutR=(psiInB*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1)/sigma)/epsilon; // find out going angular flux
			psiA3=2*(psiInB*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon)/sigma)/epsilon_2; // find cell angular flux
			
			psiA=((psiInL+psiInB)*(epsilon+exp_epsilon-1.0)+2.0*SA*(1.0+0.5*epsilon_2-exp_epsilon-epsilon)/sigma)/epsilon_2;
		}
		
		// total
		//psiOutT=psiOut1;
		//psiOutR=psiOut3;
		//psiA=0.5*(psiA1+psiA3);
		
		//cout<<"balance 1"<<endl;
		//cout<<"T1 balance "<<SA*du/mup-2*(psiOutT-psiInL)-epsilon*psiA1<<endl; // triangle 1 balance
		//cout<<"T3 balance "<<SA*du/mup-2*(psiOutR-psiInB)-epsilon*psiA3<<endl; // triangle 3 balance
		//cout<<"cell angular balance "<<SA*LT*LR-mut*LR*(psiOutR-psiInL)-etat*LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
		//cout<<"cell ang bal "<<SA*LT*LR-LR*(psiOutR-psiInL)-LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
		//cout<<SA*du-mup*(psiOutR-psiInL)-mup*(psiOutT-psiInB)-sigma*du*psiA<<endl;
	}
	else if ( mu<muc ) { // ray splits the top and bottom of the cell
		Lout1=mu*LR/sqrt(1.0-mu*mu);
		du=Lout1/mu;
		A1=Lout1*LR/2.0; // Triangle 1 Area
		Lout2=LT-Lout1;
		A2=LT*LR-2*A1; // Parallelogram 2 Area
		if ( sigma<1e-10 ) {
			// triangle A
			psiOut1=psiInL+SA*du/mup/2; // find out going angular flux
			psiA1  =psiInL+SA*du/mup/3; // find cell angular flux
			
			// parallelogram B
			psiOut2=psiInB+SA*du/mup;   // find out going angular flux
			psiA2  =psiInB+SA*du/mup/2; // find cell angular flux
			
			// triangle C
			psiOutR=psiA2;              // find out going angular flux
			psiA3  =psiInB+SA*du/mup/3; // find cell angular flux
		}
		else {
			epsilon=sigma*du/mup; // optical thickness
			exp_epsilon=exp(-epsilon); // exponent of epsilon
			epsilon_2=epsilon*epsilon; // square of epsilon
			// triangle A
			psiOut1=(psiInL*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1)/sigma)/epsilon; // find out going angular flux
			psiA1=2*(psiInL*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon)/sigma)/epsilon_2; // find cell angular flux
			
			// parallelogram B
			psiOut2=psiInB*exp_epsilon+SA*(1-exp_epsilon)/sigma;                       // find out going angular flux
			psiA2  =(psiInB*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1)/sigma)/epsilon; // find cell angular flux
			
			// triangle C
			psiOutR=psiA2;                                                                                       // find out going angular flux
			psiA3  =2*(psiInB*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon)/sigma)/epsilon_2; // find cell angular flux
			
		}
		
		// total
		psiOutT=(Lout1*psiOut1+Lout2*psiOut2)/LT;
		psiA=(A1*psiA1+A2*psiA2+A1*psiA3)/(LT*LR);
		
		//cout<<"balance 2"<<endl;
		//cout<<"T1 balance "<<SA*du/mup-2*(psiOut1-psiInL)-epsilon*psiA1<<endl; // triangle 1 balance
		//cout<<"P2 balance "<<SA*du/mup-(psiOut2-psiInB)-epsilon*psiA2<<endl; // parallelogram 2 balance
		//cout<<"T3 balance "<<SA*du/mup-2*(psiOutR-psiInB)-epsilon*psiA3<<endl; // triangle 3 balance
		//cout<<"cell angular balance "<<SA*LT*LR-mut*LR*(psiOutR-psiInL)-etat*LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
	}
	else { // ray splits the right and left side of the cell
		du=LT/mu;
		A1=sqrt(du*du-LT*LT)*LT/2.0; // Triangle 1 Area
		Lout3=sqrt(du*du-LT*LT); // Triangle 3 Length
		Lout2=LR-Lout3;
		A2=LT*LR-2*A1; // Parallelogram 2 Area
		
		if ( sigma<1e-10 ) {
			// triangle A
			psiOutT=psiInL+SA*du/mup/2; // find out going angular flux
			psiA1  =psiInL+SA*du/mup/3; // find cell angular flux
			
			// parallelogram B
			psiOut2=psiInL+SA*du/mup; // find out going angular flux
			psiA2  =psiOutT;          // find cell angular flux
			
			// triangle C
			psiOut3=psiInB+SA*du/mup/2; // find out going angular flux
			psiA3  =psiInB+SA*du/mup/2; // find cell angular flux
			
		}
		else {
			epsilon=sigma*du/mup; // optical thickness
			exp_epsilon=exp(-epsilon); // exponent of epsilon
			epsilon_2=epsilon*epsilon; // square of epsilon
			// triangle A
			psiOutT=(psiInL*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1)/sigma)/epsilon; // find out going angular flux
			psiA1=2*(psiInL*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon)/sigma)/epsilon_2; // find cell angular flux
			
			// parallelogram B
			psiOut2=psiInL*exp_epsilon+SA*(1-exp_epsilon)/sigma; // find out going angular flux
			psiA2=psiOutT;                                       // find cell angular flux
			
			// triangle C
			psiOut3=(psiInB*(1-exp_epsilon)+SA*(epsilon+exp_epsilon-1)/sigma)/epsilon; // find out going angular flux
			psiA3=2*(psiInB*(epsilon+exp_epsilon-1)+SA*(1+0.5*epsilon_2-exp_epsilon-epsilon)/sigma)/epsilon_2; // find cell angular flux
		}
		
		// total
		psiOutR=(Lout3*psiOut3+Lout2*psiOut2)/LR;
		psiA=(A1*psiA1+A2*psiA2+A1*psiA3)/(LT*LR);
		
		//cout<<"balance 3"<<endl;
		//cout<<"T1 balance "<<SA*du/mup-2*(psiOutT-psiInL)-epsilon*psiA1<<endl; // triangle 1 balance
		//cout<<"P2 balance "<<SA*du/mup-(psiOut2-psiInL)-epsilon*psiA2<<endl; // parallelogram 2 balance
		//cout<<"T3 balance "<<SA*du/mup-2*(psiOut3-psiInB)-epsilon*psiA3<<endl; // triangle 3 balance
		//cout<<"cell angular balance "<<SA*LT*LR-mut*LR*(psiOutR-psiInL)-etat*LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
	}
	//cout<<"cell angular balance "<<SA*LT*LR-mut*LR*(psiOutR-psiInL)-etat*LT*(psiOutT-psiInB)-LT*LR*sigma*psiA<<endl;
}
//======================================================================================//

//======================================================================================//
//++ function to solve LOQD problem ++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void LOQDsolution()
{
	int i, j, p, c, N_ukn, N_c, N_xf, N_yf;
	double res=0;
	
	cout<<"LOQD Solution Started : ";
	temfile<<"LOQD Solution\n";
	
	N_c=nx*ny;
	N_xf=nx*ny+ny;
	N_yf=nx*ny+nx;
	N_ukn=3*nx*ny+nx+ny;
	
	VectorXd x(N_ukn);
	VectorXd b(N_ukn);
    SpMat A(N_ukn,N_ukn);
	
	// Assign matrix A and vector b
	// Balance equations ++++++++++++++++++++++++++++++++++++++++
	p=0;
	for (j=0; j<ny; j++) {
		for (i=0; i<nx; i++) {
			A.insert(p,p) = 4*(hy[j]*E_xx[i][j]/hx[i]+hx[i]*E_yy[i][j]/hy[j])+sigmaT[i][j]*hx[i]*hy[j]*(sigmaT[i][j]-sigmaS[i][j]); // cell centre flux
			A.insert(p,N_c+p+j)    = -2*hy[j]*E_xx_x[i][j]/hx[i];  A.insert(p,N_c+p+j+1)     = -2*hy[j]*E_xx_x[i+1][j]/hx[i]; // x grid flux
			A.insert(p,N_c+N_xf+p) = -2*hx[i]*E_yy_y[i][j]/hy[j];  A.insert(p,N_c+N_xf+p+nx) = -2*hx[i]*E_yy_y[i][j+1]/hy[j]; // y grid flux
			b[p] = sigmaT[i][j]*s_ext[i][j]*hx[i]*hy[j]; // right hand side
			p++;
		}
	}
	
	// X grid equations ++++++++++++++++++++++++++++++++++++++++
	c=0;
	for (j=0; j<ny; j++) {
		// Left BC
		A.insert(p,c)          = hy[j]*E_xx[0][j]; // cell centre flux
		A.insert(p,N_c+c+j)    = hy[j]*(-E_xx_x[0][j]+0.5*sigmaT[0][j]*hx[0]*cL[j]); // x grid flux
		A.insert(p,N_c+N_xf+c) = -0.5*hx[0]*E_xy_y[0][j];  A.insert(p,N_c+N_xf+c+nx) = 0.5*hx[0]*E_xy_y[0][j+1]; // y grid flux
		b[p] = 0.5*sigmaT[0][j]*hx[0]*hy[j]*(cL[j]*phiInL[j]-jInL[j]); // right hand side
		p++; c++;
		for (i=1; i<nx; i++) {
			A.insert(p,c-1) = -2*hy[j]*E_xx[i-1][j]*sigmaT[i][j]/hx[i-1];  A.insert(p,c) = -2*hy[j]*E_xx[i][j]*sigmaT[i-1][j]/hx[i]; // center flux
			A.insert(p,N_c+c+j)      = 2*hy[j]*E_xx_x[i][j]*(sigmaT[i][j]/hx[i-1]+sigmaT[i-1][j]/hx[i]);     // x grid flux
			A.insert(p,N_c+N_xf+c-1) = -E_xy_y[i-1][j]*sigmaT[i][j];  A.insert(p,N_c+N_xf+c-1+nx) = E_xy_y[i-1][j+1]*sigmaT[i][j];  // y grid flux left
			A.insert(p,N_c+N_xf+c)   = E_xy_y[i][j]*sigmaT[i-1][j];   A.insert(p,N_c+N_xf+c+nx)   = -E_xy_y[i][j+1]*sigmaT[i-1][j]; // y grid flux right
			b[p] = 0.0; // right hand side
			p++; c++;
		}
		// Right BC
		A.insert(p,c-1)          = -hy[j]*E_xx[nx-1][j]; // cell centre flux
		A.insert(p,N_c+c+j)      = hy[j]*(E_xx_x[nx][j]+0.5*sigmaT[nx-1][j]*hx[nx-1]*cR[j]); // x grid flux
		A.insert(p,N_c+N_xf+c-1) = -0.5*hx[nx-1]*E_xy_y[nx-1][j];  A.insert(p,N_c+N_xf+c+nx-1) = 0.5*hx[nx-1]*E_xy_y[nx-1][j+1]; // y grid flux
		b[p] = 0.5*sigmaT[nx-1][j]*hx[nx-1]*hy[j]*(cR[j]*phiInR[j]-jInR[j]); // right hand side
		p++;
	}
	// Y grid equations ++++++++++++++++++++++++++++++++++++++++
	c=0;
	for (i=0; i<nx; i++) {
		// Bottom BC
		A.insert(p,c)           = hx[i]*E_yy[i][0]; // cell centre flux
		A.insert(p,N_c+c)       = -0.5*hy[0]*E_xy_x[i][0];   A.insert(p,N_c+c+1) = 0.5*hy[0]*E_xy_x[i+1][0]; // x grid flux
		A.insert(p,N_c+N_xf+c)  = hx[i]*(-E_yy_y[i][0]+0.5*sigmaT[i][0]*hy[0]*cB[i]); // y grid flux
		b[p] = 0.5*sigmaT[i][0]*hx[i]*hy[0]*(cB[i]*phiInB[i]-jInB[i]); // right hand side
		p++; c++;
	}
	for (j=1; j<ny; j++) {
		for (i=0; i<nx; i++) {
			A.insert(p,c-nx) = -2*hx[i]*E_yy[i][j-1]*sigmaT[i][j]/hy[j-1];  A.insert(p,c) = -2*hx[i]*E_yy[i][j]*sigmaT[i][j-1]/hy[j]; // centre flux
			A.insert(p,N_c+c+j-nx-1) = -E_xy_x[i][j-1]*sigmaT[i][j];  A.insert(p,N_c+c+j-nx) = E_xy_x[i+1][j-1]*sigmaT[i][j];  // x grid flux left
			A.insert(p,N_c+c+j)      = E_xy_x[i][j]*sigmaT[i][j-1];   A.insert(p,N_c+c+j+1)  = -E_xy_x[i+1][j]*sigmaT[i][j-1]; // x grid flux right
			A.insert(p,N_c+N_xf+c)   = 2*hx[i]*E_yy_y[i][j]*(sigmaT[i][j]/hy[j-1]+sigmaT[i][j-1]/hy[j]);  // y grid flux
			b[p] = 0.0; // right hand side
			p++; c++;
		}
	}
	for (i=0; i<nx; i++) {
		// Top BC
		A.insert(p,c-nx)          = -hx[i]*E_yy[i][ny-1]; // cell centre flux
		A.insert(p,N_c+c-nx+ny-1) = -0.5*hy[ny-1]*E_xy_x[i][ny-1];  A.insert(p,N_c+c-nx+ny) = 0.5*hy[ny-1]*E_xy_x[i+1][ny-1]; // x grid flux
		A.insert(p,N_c+N_xf+c)    = hx[i]*(E_yy_y[i][ny]+0.5*sigmaT[i][ny-1]*hy[ny-1]*cT[i]); // y grid flux
		b[p] = 0.5*sigmaT[i][ny-1]*hx[i]*hy[ny-1]*(cT[i]*phiInT[i]-jInT[i]); // right hand side
		p++; c++;
	}
	
	// initialize solution vector guess to transport solution
	p=0;
	for (j=0; j<ny; j++) {
		for (i=0; i<nx; i++) {
			x[p]=phiL[i][j]; // cell centre flux
			p++;
		}
	}
	for (j=0; j<ny; j++) {
		for (i=0; i<nx+1; i++) {
			x[p]=phi_xL[i][j]; // x grid flux
			p++;
		}
	}
	for (j=0; j<ny+1; j++) {
		for (i=0; i<nx; i++) {
			x[p]=phi_yL[i][j]; // y grid flux
			p++;
		}
	}
	
	
	A.prune(1e-17, 10);
	
	// Solve Ax=b iteratively with BiCGSTAB
	t_pc=clock(); // start low-order timer
	BiCGSTAB<SpMat,IncompleteLUT<double> > solver;
	solver.preconditioner().setFillfactor(11);
	solver.setTolerance(epsilon_lo);     // set convergence criteria 
	solver.setMaxIterations(maxiter_lo); // set the max number of lo iterations
	solver.compute(A);
	t_pc=clock()-t_pc; // stop low-order timer
	dt_pc.push_back(((double)t_pc)/CLOCKS_PER_SEC); // add high-order solution time to vector
	
	x = solver.solveWithGuess(b,x);
	err_lo.push_back(solver.error());      // error in lo solution
	num_lo.push_back(solver.iterations()); // number of lo iterations
	
	
	// set solution back to problem values
	p=0;
	for (j=0; j<ny; j++) {
		for (i=0; i<nx; i++) {
			phi[i][j]=x[p]; // cell centre flux
			p++;
		}
	}
	for (j=0; j<ny; j++) {
		for (i=0; i<nx+1; i++) {
			phi_x[i][j]=x[p]; // x grid flux
			p++;
		}
	}
	for (j=0; j<ny+1; j++) {
		for (i=0; i<nx; i++) {
			phi_y[i][j]=x[p]; // y grid flux
			p++;
		}
	}
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Calculate current ///////////////////////////////////////////////////////////////////////////////////////
	// Boundary currents ///////////////////////////////////////////////////////////////////////////////////////
	for (i=0; i<nx; i++) {
		j_y[i][0]=cB[i]*(phi_y[i][0]-phiInB[i])+jInB[i]; // Bottom boundary current J_y
	}
	for (j=0; j<ny; j++) {
		j_x[0][j]=cL[j]*(phi_x[0][j]-phiInL[j])+jInL[j]; // Left boundary current J_x
	}
	
	// Inner currents J_x J_y
	for (j=0; j<ny; j++) {
		for (i=0; i<nx; i++) {
			j_x[i+1][j]=-(2*hy[j]*(E_xx_x[i+1][j]*phi_x[i+1][j]-E_xx[i][j]*phi[i][j])+
				hx[i]*(E_xy_y[i][j+1]*phi_y[i][j+1]-E_xy_y[i][j]*phi_y[i][j]))/(sigmaT[i][j]*hx[i]*hy[j]);
				
			j_y[i][j+1]=-(2*hx[i]*(E_yy_y[i][j+1]*phi_y[i][j+1]-E_yy[i][j]*phi[i][j])+
				hy[j]*(E_xy_x[i+1][j]*phi_x[i+1][j]-E_xy_x[i][j]*phi_x[i][j]))/(sigmaT[i][j]*hx[i]*hy[j]);
		}
	}
	
	// Boundary currents
	for (i=0; i<nx; i++) {
		j_y[i][ny]=cT[i]*(phi_y[i][ny]-phiInT[i])+jInT[i]; // Top boundary current
	}
	for (j=0; j<ny; j++) {
		j_x[nx][j]=cR[j]*(phi_x[ny][j]-phiInR[j])+jInR[j]; // Right boundary current
	}
	cout<<"Completed \n";
}
//======================================================================================//

//======================================================================================//
//++ function to output data +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//
//======================================================================================//
void output(char fn[])
{
	char outf[25]={""};
	int i, j, m, outw=16;
	double phi_sum=0.0, psi_max=0.0, res;
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// Matrix Residuals //////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// left BC residuals
	for (j=0; j<ny; j++) {
		// left BC residuals
		res=abs(hy[j]*E_xx[0][j]*phi[0][j]+
			(-hy[j]*E_xx_x[0][j]+0.5*sigmaT[0][j]*hx[0]*hy[j]*cL[j])*phi_x[0][j]+
			0.5*hx[0]*(-E_xy_y[0][j]*phi_y[0][j]+E_xy_y[0][j+1]*phi_y[0][j+1]-
			sigmaT[0][j]*hy[j]*(cL[j]*phiInL[j]-jInL[j])));
		if ( res>res_ml ) {
			res_ml=res; j_ml=j;
		}
		// right BC residuals
		res=abs(-hy[j]*E_xx[nx-1][j]*phi[nx-1][j]+
			(hy[j]*E_xx_x[nx][j]+0.5*sigmaT[nx-1][j]*hx[nx-1]*hy[j]*cR[j])*phi_x[nx][j]+
			0.5*hx[nx-1]*(-E_xy_y[nx-1][j]*phi_y[nx-1][j]+E_xy_y[nx-1][j+1]*phi_y[nx-1][j+1]-
			sigmaT[nx-1][j]*hy[j]*(cR[j]*phiInR[j]-jInR[j])));
		if ( res>res_mr ) {
			res_mr=res; j_mr=j;
		}
	}
	for (i=0; i<nx; i++) {
		// bottom BC residuals
		res=abs(hx[i]*E_yy[i][0]*phi[i][0]+
			(-hx[i]*E_yy_y[i][0]+0.5*sigmaT[i][0]*hx[i]*hy[0]*cB[i])*phi_y[i][0]+
			0.5*hy[0]*(-E_xy_x[i][0]*phi_x[i][0]+E_xy_x[i+1][0]*phi_x[i+1][0]-
			sigmaT[i][0]*hx[i]*(cB[i]*phiInB[i]-jInB[i])));
		if ( res>res_mb ) {
			res_mb=res; i_mb=i;
		}
		// top BC residuals
		res=abs(-hx[i]*E_yy[i][ny-1]*phi[i][ny-1]+
			(hx[i]*E_yy_y[i][ny]+0.5*sigmaT[i][ny-1]*hx[i]*hy[ny-1]*cT[i])*phi_y[i][ny]+
			0.5*hy[ny-1]*(-E_xy_x[i][ny-1]*phi_x[i][ny-1]+E_xy_x[i+1][ny-1]*phi_x[i+1][ny-1]-
			sigmaT[i][ny-1]*hx[i]*(cT[i]*phiInT[i]-jInT[i])));
		if ( res>res_mt ) {
			res_mt=res; i_mt=i;
		}
	}
	// balance residuals
	for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
			res=abs((-2*hy[j]/hx[i]/sigmaT[i][j])*(E_xx_x[i+1][j]*phi_x[i+1][j]-2*E_xx[i][j]*phi[i][j]+E_xx_x[i][j]*phi_x[i][j])+
				(-2*hx[i]/hy[j]/sigmaT[i][j])*(E_yy_y[i][j+1]*phi_y[i][j+1]-2*E_yy[i][j]*phi[i][j]+E_yy_y[i][j]*phi_y[i][j])+
				(sigmaT[i][j]-sigmaS[i][j])*phi[i][j]*hx[i]*hy[j]-
				s_ext[i][j]*hx[i]*hy[j]);
			if ( res>res_mbal ) {
				res_mbal=res; i_mbal=i; j_mbal=j;
			}
		}
	}
	// x grid residuals
	for (i=1; i<nx; i++) {
		for (j=0; j<ny; j++) {
			res=abs(2*hy[j]*((-E_xx[i-1][j]/(sigmaT[i-1][j]*hx[i-1]))*phi[i-1][j]+
				(-E_xx[i][j]  /(sigmaT[i][j]  *hx[i]  ))*phi[i][j]+
				(E_xx_x[i][j]*(1/(sigmaT[i-1][j]*hx[i-1])+1/(sigmaT[i][j]*hx[i])))*phi_x[i][j])+
				(-E_xy_y[i-1][j] /sigmaT[i-1][j])*phi_y[i-1][j]+(E_xy_y[i-1][j+1]/sigmaT[i-1][j])*phi_y[i-1][j+1]+
				(E_xy_y[i][j]   /sigmaT[i][j])*phi_y[i][j]+(-E_xy_y[i][j+1]/sigmaT[i][j])*phi_y[i][j+1]);
			if ( res>res_mx ) {
				res_mx=res; i_mx=i; j_mx=j;
			}
		}
	}
	// y grid residuals
	for (j=1; j<ny; j++) {
		for (i=0; i<nx; i++) {
			res=abs(2*hx[i]*((-E_yy[i][j-1]/(sigmaT[i][j-1]*hy[j-1]))*phi[i][j-1]+
				(-E_yy[i][j]  /(sigmaT[i][j]  *hy[j]  ))*phi[i][j]+
				(E_yy_y[i][j]*(1/(sigmaT[i][j-1]*hy[j-1])+1/(sigmaT[i][j]*hy[j])))*phi_y[i][j])+
				(-E_xy_x[i][j-1] /sigmaT[i][j-1])*phi_x[i][j-1]+(E_xy_x[i+1][j-1]/sigmaT[i][j-1])*phi_x[i+1][j-1]+
				( E_xy_x[i][j]  /sigmaT[i][j])*phi_x[i][j]+(-E_xy_x[i+1][j]/sigmaT[i][j])*phi_x[i+1][j]);
			if ( res>res_my ) {
				res_my=res; i_my=i; j_my=j;
			}
		}
	}
	
	// calculate residuals ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// Equation residuals
	res_bal=0;
	res_l=0; res_r=0;
	res_b=0; res_t=0;
	for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) {
			// Balance in Cell
			res=abs((j_x[i+1][j]-j_x[i][j])*hy[j]+(j_y[i][j+1]-j_y[i][j])*hx[i]+
			(sigmaT[i][j]-sigmaS[i][j])*phi[i][j]*hx[i]*hy[j]-s_ext[i][j]*hx[i]*hy[j]);
			if ( res>res_bal ) {
				res_bal=res;
				i_bal=i; j_bal=j;
			}
			// Left Half of Cell
			res=abs(2*hy[j]*(E_xx[i][j]*phi[i][j]-E_xx_x[i][j]*phi_x[i][j])+
			+hx[i]*(E_xy_y[i][j+1]*phi_y[i][j+1]-E_xy_y[i][j]*phi_y[i][j])+sigmaT[i][j]*hx[i]*hy[j]*j_x[i][j]);
			if ( res>res_l ) {
				res_l=res;
				i_l=i; j_l=j;
			}
			// Right Half of Cell
			res=abs(2*hy[j]*(E_xx_x[i+1][j]*phi_x[i+1][j]-E_xx[i][j]*phi[i][j])+
			+hx[i]*(E_xy_y[i][j+1]*phi_y[i][j+1]-E_xy_y[i][j]*phi_y[i][j])+sigmaT[i][j]*hx[i]*hy[j]*j_x[i+1][j]);
			if ( res>res_r ) {
				res_r=res;
				i_r=i; j_r=j;
			}
			// Bottom Half of Cell
			res=abs(2*hx[i]*(E_yy[i][j]*phi[i][j]-E_yy_y[i][j]*phi_y[i][j])+
			+hy[j]*(E_xy_x[i+1][j]*phi_x[i+1][j]-E_xy_x[i][j]*phi_x[i][j])+sigmaT[i][j]*hx[i]*hy[j]*j_y[i][j]);
			if ( res>res_b ) {
				res_b=res;
				i_b=i; j_b=j;
			}
			// Top Half of Cell
			res=abs(2*hx[i]*(E_yy_y[i][j+1]*phi_y[i][j+1]-E_yy[i][j]*phi[i][j])+
			+hy[j]*(E_xy_x[i+1][j]*phi_x[i+1][j]-E_xy_x[i][j]*phi_x[i][j])+sigmaT[i][j]*hx[i]*hy[j]*j_y[i][j+1]);
			if ( res>res_t ) {
				res_t=res;
				i_t=i; j_t=j;
			}
		}
	}
	
	res_lbc=0; res_rbc=0;
	res_bbc=0; res_tbc=0;
	for (j=0; j<ny; j++) { // Left and Right
		res=abs(cL[j]*(phi_x[0][j]-phiInL[j])+jInL[j]-j_x[0][j]); // Left BC residual
		if ( res>res_lbc ) {
			res_lbc=res;
			j_lbc=j;
		}
		res=abs(cR[j]*(phi_x[nx][j]-phiInR[j])+jInR[j]-j_x[nx][j]); // Right BC residual
		if ( res>res_rbc ) {
			res_rbc=res;
			j_rbc=j;
		}
	}
	for (i=0; i<nx; i++) { // Bottom and Top
		res=abs(cB[i]*(phi_y[i][0]-phiInB[i])+jInB[i]-j_y[i][0]); // Bottom BC residual
		if ( res>res_bbc ) {
			res_bbc=res;
			i_bbc=i;
		}
		res=abs(cT[i]*(phi_y[i][ny]-phiInT[i])+jInT[i]-j_y[i][ny]); // Top BC residual
		if ( res>res_tbc ) {
			res_tbc=res;
			i_tbc=i;
		}
	}
	
	cout.precision(8);
	// find area averaged flux
	for (i=0; i<nx; i++) {
		for (j=0; j<ny; j++) 
			phi_sum+=phi[i][j]*hx[i]*hy[j];
	}
	cout<<"Area Averaged Flux :"<<print_out(phi_sum)<<"-------"<<endl;
	
	
	// file output ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	// boundary conditions
	outfile<<"\n -- Boundary Conditions -- \n";
	outfile<<"Type of BC: "<<kbc;   // Write type of BC
	
	
	switch ( kbc ) {
	case 1:
		outfile<<"  incoming according to side\n";
		outfile<<"Left   BC Incoming "<<bcL<<endl; // Write Left BC
		outfile<<"Bottom BC Incoming "<<bcB<<endl; // Write Bottom BC
		outfile<<"Right  BC Incoming "<<bcR<<endl; // Write Right BC
		outfile<<"Top    BC Incoming "<<bcT<<endl; // Write Top BC
		break;
	case 2:
		outfile<<"  incoming according to angle\n";
		outfile<<"Quad 1 Incoming "<<bcL<<endl; // Write Left BC
		outfile<<"Quad 2 Incoming "<<bcB<<endl; // Write Bottom BC
		outfile<<"Quad 3 Incoming "<<bcR<<endl; // Write Right BC
		outfile<<"Quad 4 Incoming "<<bcT<<endl; // Write Top BC
		break;
	case 3:
		outfile<<"  reflective on all sides\n";
		outfile<<"Left   BC Reflective\n"; // Write Left BC
		outfile<<"Bottom BC Reflective\n"; // Write Bottom BC
		outfile<<"Right  BC Reflective\n"; // Write Right BC
		outfile<<"Top    BC Reflective\n"; // Write Top BC
		break;
	case 4:
		outfile<<"  reflective on Left and Bottom, incoming on Right and Top according to side\n";
		outfile<<"Left   BC Reflective\n"; // Write Left BC
		outfile<<"Bottom BC Reflective\n"; // Write Bottom BC
		outfile<<"Right  BC Incoming "<<bcR<<endl; // Write Right BC
		outfile<<"Top    BC Incoming "<<bcT<<endl; // Write Top BC
		break;
	case 5:
		outfile<<"  reflective on Right and Top, incoming on Left and Bottom according to side\n";
		outfile<<"Left   BC Incoming "<<bcL<<endl; // Write Left BC
		outfile<<"Bottom BC Incoming "<<bcB<<endl; // Write Bottom BC
		outfile<<"Right  BC Reflective\n";; // Write Right BC
		outfile<<"Top    BC Reflective\n";; // Write Top BC
		break;
	default:
		cout<<"Invalid BC Type !!!!!\n";
		break;
	}
	
	// output grid
	outfile<<"\n -- Solution Grid -- \n";
	outfile<<" X grid \n"<<" index  cell edge cell centre \n";
	for (i=0; i<nx; i++) outfile<<setw(6)<<i+1<<setw(11)<<x[i]<<setw(11)<<(x[i]+x[i+1])/2<<endl; // write x grid
	outfile<<setw(6)<<nx+1<<setw(11)<<x[nx]<<endl;
	outfile<<" Y grid \n"<<" index  cell edge cell centre \n";
	for (j=0; j<ny; j++) outfile<<setw(6)<<j+1<<setw(11)<<y[j]<<setw(11)<<(y[j]+y[j+1])/2<<endl; // write y grid
	outfile<<setw(6)<<ny+1<<setw(11)<<y[ny]<<endl;
	
	// output quadrature
	outfile<<"\n -- Quadrature -- \n Level Symmetric S"<<N<<" Normalized to Integrate to 4*pi \n";
	outfile<<"  m      mu        eta        xi      weight  \n";
	for (m=0; m<sn; m++) outfile<<setw(5)<<m+1<<setw(10)<<mu[m]<<setw(10)<<eta[m]<<setw(10)<<xi[m]<<setw(10)<<w[m]<<endl; // quadrature
	outfile<<endl;
	
	outfile<<endl;
	outfile<<" -------------- \n";
	outfile<<" -- Solution -- \n";
	outfile<<" -------------- \n";
	outfile<<"Program run time : "<<((float)t)/CLOCKS_PER_SEC<<" seconds"<<endl;
	outfile<<"\nNumber of iterations "<<n_iterations<<endl;
	
	outfile<<"\n -- Iteration Data -- \n";
	outfile<<"  Iter.   Convergence       # LO         LO Solution     High-order       Low-order    Preconditioner\n";
	outfile<<"    #        Rate        Iterations      Matrix Error   Sol. Time [s]   Sol. Time [s]  Sol. Time [s]\n";
	for (i=0; i<n_iterations; i++) {
		outfile<<setw(4)<<i+1<<setw(2)<<" "<<print_out(rho_si[i+1])<<
		setw(outw/2)<<num_lo[i]+1<<setw(outw/2-1)<<" "<<print_out(err_lo[i+1])<<
		print_out(dt_ho[i])<<print_out(dt_lo[i])<<print_out(dt_pc[i+1])<<endl;
	}
	
	
	outfile<<"\n -- Residuals -- \n";
	outfile<<"High-order Residual \n";
	outfile<<"Balance Residual:"<<print_out(max_res)<<" at "<<ii<<" , "<<jj<<endl;
	
	outfile<<"Low-order Residuals \n";
	outfile<<"Residual of Equations Solved In Matrix \n";
	outfile<<"Cell Balance Residual:"<<print_out(res_mbal)<<" at "<<i_mbal<<" , "<<j_mbal<<endl; 
	outfile<<"X Grid       Residual:"<<print_out(res_mx)<<" at "<<i_mx<<" , "<<j_mx<<endl;
	outfile<<"Y Grid       Residual:"<<print_out(res_my)<<" at "<<i_my<<" , "<<j_my<<endl;
	outfile<<"Left    BC   Residual:"<<print_out(res_ml)<<" at "<<j_ml<<endl;
	outfile<<"Right   BC   Residual:"<<print_out(res_mr)<<" at "<<j_mr<<endl;
	outfile<<"Bottom  BC   Residual:"<<print_out(res_mb)<<" at "<<i_mb<<endl;
	outfile<<"Top     BC   Residual:"<<print_out(res_mt)<<" at "<<i_mt<<endl;
	
	outfile<<"Residual of General Equations \n";
	outfile<<"Cell Balance Residual:"<<print_out(res_bal)<<" at "<<i_bal<<" , "<<j_bal<<endl; 
	outfile<<"Left    Cell Residual:"<<print_out(res_l)<<" at "<<i_l<<" , "<<j_l<<endl;
	outfile<<"Right   Cell Residual:"<<print_out(res_r)<<" at "<<i_r<<" , "<<j_r<<endl;
	outfile<<"Bottom  Cell Residual:"<<print_out(res_b)<<" at "<<i_b<<" , "<<j_b<<endl;
	outfile<<"Top     Cell Residual:"<<print_out(res_t)<<" at "<<i_t<<" , "<<j_t<<endl;
	outfile<<"Left    BC   Residual:"<<print_out(res_lbc)<<" at "<<j_lbc<<endl;
	outfile<<"Right   BC   Residual:"<<print_out(res_rbc)<<" at "<<j_rbc<<endl;
	outfile<<"Bottom  BC   Residual:"<<print_out(res_bbc)<<" at "<<i_bbc<<endl;
	outfile<<"Top     BC   Residual:"<<print_out(res_tbc)<<" at "<<i_tbc<<endl;
	
	// output flux
	outfile<<"\n ------------------- ";
	outfile<<"\n -- LOQD Solution -- "; // Write LOQD solution
	outfile<<"\n ------------------- \n";
	outfile<<"\n -- Cell Averaged Scalar Flux -- \n";
	write_cell_average_out(phi, outw, outfile); // call function to write out cell average scalar flux
	
	outfile<<"\n -- X Grid Face Average Scalar Flux -- \n";
	write_cell_edge_x_out(phi_x, outw, outfile); // call function to write out cell edge scalar flux on x grid
	
	outfile<<"\n -- Y Grid Face Average Scalar Flux -- \n";
	write_cell_edge_y_out(phi_y, outw, outfile); // call function to write out cell edge scalar flux on y grid
	
	outfile<<"\n -- X Face Average Normal Current J_x -- \n";
	write_cell_edge_x_out(j_x, outw, outfile); // call function to write out cell edge current on x grid
	
	outfile<<"\n -- Y Face Average Normal Current J_y -- \n";
	write_cell_edge_y_out(j_y, outw, outfile); // call function to write out cell edge scalar flux on y grid
	//////////////////////////////////////////////////////////////////////////////////////////////////
	outfile<<"\n ------------------------- ";
	outfile<<"\n -- High Order Solution -- "; // Write Step Characteristics solution
	outfile<<"\n ------------------------- \n";
	outfile<<"\n -- Cell Averaged Scalar Flux -- \n";
	write_cell_average_out(phiT, outw, outfile); // call function to write out cell average scalar flux
	
	outfile<<"\n -- X Vertical Cell Edge Scalar Flux -- \n";
	write_cell_edge_x_out(phi_xT, outw, outfile); // call function to write out cell edge scalar flux on x grid
	
	outfile<<"\n -- Y Horizontal Cell Edge Scalar Flux -- \n";
	write_cell_edge_y_out(phi_yT, outw, outfile); // call function to write out cell edge scalar flux on y grid
	
	outfile<<"\n -- X Face Average Normal Current J_x -- \n";
	write_cell_edge_x_out(j_xT, outw, outfile); // call function to write out cell edge current on x grid
	
	outfile<<"\n -- Y Face Average Normal Current J_y -- \n";
	write_cell_edge_y_out(j_yT, outw, outfile); // call function to write out cell edge scalar flux on y grid
	
	outfile<<"\n -- Cell Average E_xx Eddington Tensor -- \n";
	write_cell_average_out(E_xx, outw, outfile); // call function to write out cell average Eddington Tensor E_xx
	
	outfile<<"\n -- Cell Average E_yy Eddington Tensor -- \n";
	write_cell_average_out(E_yy, outw, outfile); // call function to write out cell average Eddington Tensor E_yy
	
	outfile<<"\n -- Cell Average E_xy Eddington Tensor -- \n";
	write_cell_average_out(E_xy, outw, outfile); // call function to write out cell average Eddington Tensor E_xy
	
	outfile<<"\n -- X Vertical Cell Edge E_xx Eddington Tensor -- \n";
	write_cell_edge_x_out(E_xx_x, outw, outfile); // call function to write out cell edge Eddington Tensor E_xx on x grid
	
	outfile<<"\n -- X Vertical Cell Edge E_yy Eddington Tensor -- \n";
	write_cell_edge_x_out(E_yy_x, outw, outfile); // call function to write out cell edge Eddington Tensor E_yy on x grid
	
	outfile<<"\n -- X Vertical Cell Edge E_xy Eddington Tensor -- \n";
	write_cell_edge_x_out(E_xy_x, outw, outfile); // call function to write out cell edge Eddington Tensor E_xy on x grid
	
	outfile<<"\n -- Y Horizontal Cell Edge E_xx Eddington Tensor -- \n";
	write_cell_edge_y_out(E_xx_y, outw, outfile); // call function to write out cell edge Eddington Tensor E_xx on y grid
	
	outfile<<"\n -- Y Horizontal Cell Edge E_yy Eddington Tensor -- \n";
	write_cell_edge_y_out(E_yy_y, outw, outfile); // call function to write out cell edge Eddington Tensor E_yy on y grid
	
	outfile<<"\n -- Y Horizontal Cell Edge E_xy Eddington Tensor -- \n";
	write_cell_edge_y_out(E_xy_y, outw, outfile); // call function to write out cell edge Eddington Tensor E_xy on y grid
	
	// Boundary conditions
	outfile<<" -- Bottom and Top Boundary Factors -- \n";
	outfile<<" index "<<"     x centre    "<<"  J In Bottom   "<<" Flux In Bottom "<<"    C Bottom    ";
	outfile<<"    J In Top    "<<"  Flux In Top   "<<"     C Top      \n";
	for (i=0; i<nx; i++) {
		outfile<<setw(6)<<i<<" "<<print_out((x[i]+x[i+1])/2)<<
		print_out(jInB[i])<<print_out(phiInB[i])<<print_out(cB[i])<<
		print_out(jInT[i])<<print_out(phiInT[i])<<print_out(cT[i])<<endl;
	}
	
	outfile<<" -- Left and Right Boundary Factors -- \n";
	outfile<<" index "<<"     y centre    "<<"   J In Left    "<<"  Flux In Left  "<<"     C Left     ";
	outfile<<"   J In Right   "<<" Flux In Right  "<<"    C Right     \n";
	for (j=0; j<ny; j++) {
		outfile<<setw(6)<<i<<" "<<print_out((y[j]+y[j+1])/2)<<
		print_out(jInL[j])<<print_out(phiInL[j])<<print_out(cL[j])<<
		print_out(jInR[j])<<print_out(phiInR[j])<<print_out(cR[j])<<endl;
	}
	
	outfile.close(); // close output file opened in input file +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	if ( nx<500 && ny<500 ) {
		strcat (outf,fn); strcat (outf,".csv"); // name output file
		cout<<outf<<endl;
		datfile.open(outf); // open output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		// output flux
		datfile<<"# of iterations ,"<<n_iterations<<endl;
		datfile<<" -- Iteration Data -- \n";
		datfile<<"  Iter.,   Convergence,       # LO    ,   LO Solution ,  High-order   ,   Low-order   ,\n";
		datfile<<"    #  ,      Rate    ,    Iterations ,   Matrix Error, Sol. Time [s] ,  Sol. Time [s],\n";
		for (i=0; i<n_iterations; i++) {
			datfile<<setw(4)<<i+1<<setw(2)<<" "<<","<<print_csv(rho_si[i+1])
			<<setw(outw/2)<<num_lo[i]+1<<setw(outw/2-1)<<" "<<","<<print_csv(err_lo[i])<<
			print_csv(dt_ho[i])<<print_csv(dt_lo[i])<<endl;
			//print_csv(dt_ho[i])<<print_csv(dt_lo[i])<<endl;
		}
		datfile<<" # of x cells , # of y cells ,\n";
		datfile<<nx<<" , "<<ny<<", \n";
		
		
		// output grid
		datfile<<" -- Solution Grid -- \n";
		datfile<<" X grid \n"<<" index , cell edge , cell centre ,\n";
		for (i=0; i<nx; i++) {
			datfile<<setw(6)<<i+1<<","<<print_csv(x[i])<<print_csv((x[i]+x[i+1])/2)<<endl; // write x grid
		}
		datfile<<setw(6)<<nx+1<<","<<print_csv(x[nx])<<endl;
		
		datfile<<" Y grid \n"<<" index , cell edge , cell centre ,\n";
		for (j=0; j<ny; j++) {
			datfile<<setw(6)<<j+1<<","<<print_csv(y[j])<<print_csv((y[j]+y[j+1])/2)<<endl; // write y grid
		}
		datfile<<setw(6)<<ny+1<<","<<print_csv(y[ny])<<endl;
		
		
		datfile<<" ------------------- \n";
		datfile<<" -- LOQD Solution -- \n"; // Write LOQD solution
		datfile<<" ------------------- \n";
		datfile<<" -- Cell Averaged Scalar Flux -- \n";
		write_cell_average_dat(phi, outw, datfile); // call function to write out cell average scalar flux
		
		datfile<<" -- X Grid Face Average Scalar Flux -- \n";
		write_cell_edge_x_dat(phi_x, outw, datfile); // call function to write out cell edge scalar flux on x grid
		
		datfile<<" -- Y Grid Face Average Edge Scalar Flux -- \n";
		write_cell_edge_y_dat(phi_y, outw, datfile); // call function to write out cell edge scalar flux on y grid
		
		datfile<<" -- X Face Average Normal Current J_x -- \n";
		write_cell_edge_x_dat(j_x, outw, datfile); // call function to write out cell edge current on x grid
		
		datfile<<" -- Y Face Average Normal Current J_y -- \n";
		write_cell_edge_y_dat(j_y, outw, datfile); // call function to write out cell edge current on y grid
		
		datfile<<" ------------------------- \n";
		datfile<<" -- High Order Solution -- \n"; // Write Step Characteristics solution
		datfile<<" ------------------------- \n";
		datfile<<" -- Cell Averaged Scalar Flux -- \n";
		write_cell_average_dat(phiT, outw, datfile); // call function to write out cell average scalar flux
		
		datfile<<" -- X Vertical Cell Edge Scalar Flux -- \n";
		write_cell_edge_x_dat(phi_xT, outw, datfile); // call function to write out cell edge scalar flux on x grid
		
		datfile<<" -- Y Horizontal Cell Edge Scalar Flux -- \n";
		write_cell_edge_y_dat(phi_yT, outw, datfile); // call function to write out cell edge scalar flux on y grid
		
		datfile<<" -- X Face Average Normal Current J_x -- \n";
		write_cell_edge_x_dat(j_xT, outw, datfile); // call function to write out cell edge current on x grid
		
		datfile<<" -- Y Face Average Normal Current J_y -- \n";
		write_cell_edge_y_dat(j_yT, outw, datfile); // call function to write out cell edge current on y grid
		
		datfile<<" -- Cell Average E_xx Eddington Tensor -- \n";
		write_cell_average_dat(E_xx, outw, datfile); // call function to write out cell average Eddington Tensor E_xx
		
		datfile<<" -- Cell Average E_yy Eddington Tensor -- \n";
		write_cell_average_dat(E_yy, outw, datfile); // call function to write out cell average Eddington Tensor E_yy
		
		datfile<<" -- Cell Average E_xy Eddington Tensor -- \n";
		write_cell_average_dat(E_xy, outw, datfile); // call function to write out cell average Eddington Tensor E_xy
		
		datfile<<" -- X Vertical Cell Edge Scalar E_xx Eddington Tensor -- \n";
		write_cell_edge_x_dat(E_xx_x, outw, datfile); // call function to write out cell edge Eddingtion Tensor E_xx on x grid
		
		datfile<<" -- X Vertical Cell Edge Scalar E_yy Eddington Tensor -- \n";
		write_cell_edge_x_dat(E_yy_x, outw, datfile); // call function to write out cell edge Eddingtion Tensor E_yy on x grid
		
		datfile<<" -- X Vertical Cell Edge Scalar E_xy Eddington Tensor -- \n";
		write_cell_edge_x_dat(E_xy_x, outw, datfile); // call function to write out cell edge Eddingtion Tensor E_xy on x grid
		
		datfile<<" -- Y Horizontal Cell Edge E_xx Eddington Tensor -- \n";
		write_cell_edge_y_dat(E_xx_y, outw, datfile); // call function to write out cell edge Eddingtion Tensor E_xx on y grid
		
		datfile<<" -- Y Horizontal Cell Edge E_yy Eddington Tensor -- \n";
		write_cell_edge_y_dat(E_yy_y, outw, datfile); // call function to write out cell edge Eddingtion Tensor E_yy on y grid
		
		datfile<<" -- Y Horizontal Cell Edge E_xy Eddington Tensor -- \n";
		write_cell_edge_y_dat(E_xy_y, outw, datfile); // call function to write out cell edge Eddingtion Tensor E_xy on y grid
		
		// Boundary conditions
		datfile<<" -- Bottom and Top Boundary Factors -- \n";
		datfile<<" index,"<<"    x centre   ,"<<"  J In Bottom  ,"<<" Flux In Bottom,"<<"    C Bottom   ,";
		datfile<<"    J In Top   ,"<<"  Flux In Top  ,"<<"     C Top     ,\n";
		for (i=0; i<nx; i++) {
			datfile<<setw(6)<<i<<","<<print_csv((x[i]+x[i+1])/2)
			<<print_csv(jInB[i])<<print_csv(phiInB[i])<<print_csv(cB[i])
			<<print_csv(jInT[i])<<print_csv(phiInT[i])<<print_csv(cT[i])<<endl;
		}
		
		datfile<<" -- Left and Right Boundary Factors -- \n";
		datfile<<" index,"<<"    y centre   ,"<<"   J In Left   ,"<<"  Flux In Left ,"<<"     C Left    ,";
		datfile<<"   J In Right  ,"<<" Flux In Right ,"<<"    C Right    ,\n";
		for (j=0; j<ny; j++) {
			datfile<<setw(6)<<i<<","<<print_csv((y[j]+y[j+1])/2)
			<<print_csv(jInL[j])<<print_csv(phiInL[j])<<print_csv(cL[j])
			<<print_csv(jInR[j])<<print_csv(phiInR[j])<<print_csv(cR[j])<<endl;
		}
		
		datfile.close(); // close output data file ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	}
	
}

//======================================================================================//

