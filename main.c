/*Copyright 2016 Ozan Gunaydin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.   
*/

#include <msp430xG46x.h>                         // Specidic Device Header file
#include <stdio.h>
#include <stdlib.h>
#include <intrinsics.h>				// Intrinsic functions
#include <stdint.h>			       // Integers of defined sizes
#include <time.h>
#include <math.h>


// Data Dependent settings for Neural Network
#define numInputs  4                // Neural Network inputs
#define numPatterns  1              // Input pattern Number, if training utilized it is the number of input patterns

// User defineable Neural Network settings
#define numHidden 20                // Number of hidden neurons
// const int numEpochs = 10000;     // To be used in training mode
const double LR_IH = 0.7;           // input learning rate
const double LR_HO = 0.07;          // output learning rate


// Band pass FIR filter coefficients for Beta Rhythm extraction , 19-24 Hz
static const int coeffsbp2[14] = {
     334, 1153, 775, -2639, -6095, -3382, 7289, 15549, 8170,-12686,-25777, -13271,15302,15135};

// Band pass FIR filter coefficients for Mu Rhythm extraction  ,  8 - 12 Hz
static const int coeffsbp1[12] = {
     9453, -1255, -3809, -7204,-10221, -11525, -10079, -5625, 1038,8175,13618, 7836};


#define LCDMEMS	11		// LCD memories used (3-13)
// Pointer to LCD memory used: allows use of array LCDMem[]
uint8_t * const LCDMem = (uint8_t *) &LCDM3;	
// LCD segment definitions (SoftBaugh SBLCDA4)
#define SEG_A   BIT0					//  AAAA
#define SEG_B   BIT1					// F    B
#define SEG_C   BIT2					// F    B
#define SEG_D   BIT3					//  GGGG
#define SEG_E   BIT6					// E    C
#define SEG_F   BIT4					// E    C
#define SEG_G   BIT5					//  DDDD
#define SEG_H   BIT7					// colon, point

// Patterns for hexadecimal characters
const uint8_t LCDHexChar[] = {
	SEG_A | SEG_B | SEG_C | SEG_D | SEG_E | SEG_F,			// '0'
	SEG_B | SEG_C,							// '1'
	SEG_C | SEG_F | SEG_E | SEG_G,					// 'H'
	SEG_F | SEG_E | SEG_D,					          // 'L'
	SEG_A | SEG_B | SEG_F | SEG_G | SEG_E,				// 'P'
	SEG_G | SEG_E,			                                // 'r'
	SEG_F | SEG_B | SEG_C | SEG_D | SEG_G,			        // 'y'
	SEG_F | SEG_G | SEG_E,						// 't'
	SEG_F | SEG_E | SEG_D | SEG_C | SEG_B,	                      // 'U'
	SEG_G | SEG_E | SEG_D | SEG_C,			              // 'o'
	SEG_A | SEG_B | SEG_C | SEG_E | SEG_F | SEG_G,			// 'A'
	SEG_A | SEG_F | SEG_G | SEG_C | SEG_D,			      // 'S'
	SEG_A | SEG_F | SEG_E | SEG_D | SEG_C,		              // 'g'
	SEG_B | SEG_C | SEG_D | SEG_E | SEG_G,				// 'd'
	SEG_A | SEG_D | SEG_E | SEG_F | SEG_G,				// 'E'
	SEG_A | SEG_E | SEG_F | SEG_D,					// 'C'
};
const uint8_t LCDBlankChar = 0;


// ADC variables declaration
//
  int A0results[448];    // Data array for ADC samples CH0
  int A1results[448];    // Data array for ADC samples CH1

int A0muResult[448];     // Data array for filtered signals
int A1muResult[448];      // Data array for filtered signals

// Neural Network variables 
int patNum = 0;             //Number of input pattern
double errThisPat = 0.0;    //Neural network Error
double outPred = 0.0;       // Neural network output
double RMSerror = 0.0;      // Neural Network RMS error
  // NN the outputs of the hidden neurons
double hiddenVal[numHidden];
  // NN the weights
double weightsIH[numInputs][numHidden];
double weightsHO[numHidden];
  //NN Input data
 double trainInputs[numInputs];
 
//variables for calculations
unsigned int i,Index;
int  z,clas, sum1,sum2,output1,output2 =0;
float mean_mu_C3,mean_mu_C4,mean_B_C3,mean_B_C4;    // feature variables

int compare(const void * a, const void * b)         //compare function for sorting
{
  return ( *(int*)a - *(int*)b );
}


// function prototypes
void Init(void);                               // Initializes the peripherals
void InitLCD(void);                            // Clears the LCD memory

int filterbp2(int);                     // Band Pass filter Beta Rhythm
int filterbp1(int);                      // Band Pass filter Mu Rhythm

  // LCD writing functions
void DisplayHello(void);
void DisplayResult1(void);
void DisplayResult2(void);
void DisplayReady(void);
void DisplayGo(void);
void DisplayStop(void);

  //Neural Net Functions
void initWeights();
void calcNet();
//void WeightChangesHO();            //Weight changes for training mode
//void WeightChangesIH();           //Weight changes for training mode
//void calcOverallError();          //Error Calculating for backpropagation
//void displayResults();
double getRand();


long mul16(register int x, register int y);            // 16-bit signed multiplication


// main function
void main(void)
{   Init();                               // Initialize device for the application
    DisplayHello();
    
    while(1)           // loop forever
   { _BIS_SR(LPM3_bits + GIE);       // Enter Low power mode 3, interrupts enable
    
    
    if (Index == 448)               // Stop and calculate class when 448 array filled
    {
      DisplayStop();
      TACTL =  MC_0;            // stop Timer A 
      
      BTCTL |= BIT6;            // stop basic timer
      
      for(i=0;i<60000;i++)           // delay for Stop message
      {;}
      

      
       for (i=0;i<447;i++)        //filtering to extract Mu Rhythm
      {
				output1, output2=0;
				output1=filterbp1(A0results[i]);
                                output2=filterbp1(A1results[i]);
                                 A0muResult[i]=output1;
                                 A1muResult[i]=output2;
			}
     
           for(i=0;i<447;i++)                 // take absolute value
      {
         A0muResult[i] = abs(A0muResult[i]);
         A1muResult[i] = abs(A1muResult[i]);
        } 
      
      sum1=0;
      sum2=0;
       for (i=0;i<447;i++)                 // Calculate mean values of bands
			{
				
                              //  sum1 = sum1 + A0BResult[i];
                                sum1 = sum1 +  A0muResult[i];
                              //  sum2 = sum2 + A0results[i];
                                sum2 = sum2 + A1muResult[i];
			}
      mean_mu_C3 = sum1 / 448;               // first two features: mean of absolute values
      mean_mu_C4 = sum2 / 448;
      

      
       for (i=0;i<447;i++)        //filtering to extract Mu Rhythm
{
				output1, output2=0;
				
				output1=filterbp2(A0results[i]);
                                output2=filterbp2(A1results[i]);
			
                                 A0muResult[i]=output1;
                                 A1muResult[i]=output2; 
			}      
      
      for(i=0;i<447;i++)                          // take absolute value
      {
         A0muResult[i] = abs(A0muResult[i]);
         A1muResult[i] = abs(A1muResult[i]);
      }
      

      sum1=0;
      sum2=0;
                                
      for (i=0;i<447;i++)                        // Calculate mean values of bands
			{
				
                               // sum1 = sum1 + A0BResult[i];
                                sum1 = sum1 + A0muResult[i];
                               // sum2 = sum2 + A0results[i];
                                sum2 = sum2 + A1muResult[i];
			}
      mean_B_C3 = sum1 / 448;               // last two features: mean of absolute values
      mean_B_C4 = sum2 / 448;
      
      
// Scaling values for neural network input
// Scaling is done by calculating min and max values of the arrays and then
// scaling them to the [-1,1] range.
      // Scaling formula: I = Imin + ((Imax - Imin)*(D-Dmin)/(Dmax -Dmin))
      
    trainInputs[0] = (((mean_mu_C3 - 100)/ 900)*2)-1;
    trainInputs[1] = (((mean_mu_C4 - 100)/ 900)*2)-1;
    trainInputs[2] = (((mean_B_C3 - 50)/ 400)*2)-1;
    trainInputs[3] = (((mean_B_C4 - 50)/ 400)*2)-1;
    
    calcNet();             //calculating Neural network response
      
    

            
      if (outPred < 0.0)       
      { DisplayResult1();}        // if output is negative,then result is left hand
      else if (outPred > 0.0)
      { DisplayResult2();}      // if output is positive,then result is right hand
      else 
      {if(mean_mu_C3> mean_mu_C4){DisplayResult1();} //if output is zero then compare mu rhythm means
      else{DisplayResult2();}}
      
      
       
            for(i=60000;i>0;i--)        // delay
            {;}
          //   BTCTL |= BIT6;            // stop basic timer
             
             
              for(i=60000;i>0;i--)      // delay
            {;}
      
            Index=0;    // reset index counter
            
      DisplayHello();
      
    }   // end of data processing, turn back to the infitine loop
    }  // infinite loop
}//main


// Initialization function
void Init(void)
{   
    initWeights();                      // Init Neural Network Weights
    FLL_CTL0 |= XCAP18PF;               // Set load capacitance for xtal
    WDTCTL = WDTPW | WDTHOLD;           // Disable the Watchdog
    while ( LFOF & FLL_CTL0);           // wait for watch crystal to stabilize
    SCFQCTL = 63;                       // 32 x 32768 x 2 = 2.097152MHz

    InitLCD();                          // Clear LCD memory

    P1OUT = 0x00;
    P1DIR = 0xfe;                          // Unused pins as outputs, button pins as inputs
    P2OUT = 0x00;                          // Clear P2OUT register
    P2DIR = 0xff;                          // Unused pins as outputs
    P3OUT = 0x00;                          // Clear P3OUT register
    P3DIR = 0xff;                         // Unused pins as outputs
    P4OUT = 0x00;                          // Clear P4OUT register
    P4DIR = 0xff;                          // Unused pins as outputs
    P5OUT = 0x00;                          // Clear P5OUT register
 //   P5DIR = 0xff;                        // Unused pins as outputs
    P5SEL = BIT4|BIT3|BIT2;
    
    P6OUT = 0x00;                            // Clear P6OUT register
    P6SEL = 0xff;                             // P6 = Analog selection for ADC input
    P7OUT = 0x00;
    P7DIR = 0xff;
   
    
    P1IE |= BIT0;                      //Enable button 1 interrupt
    P1IES |= BIT0;

  do {                                 // Clear interrupts for Port1
  P1IFG = 0; // Clear any pending interrupts ...
  } while (P1IFG != 0); // ... until none remain
  
// Initialize and enable ADC12
    ADC12CTL0 = ADC12ON + SHT0_8 + REFON;      // ADC12 ON, Reference = 1.5V internal, -1.5V ext
    ADC12CTL1 = SHP + SHS_1 + CONSEQ_3;        // Use sampling timer, TA1 trigger
     ADC12MCTL0 = INCH_0 + SREF_5 ;           // A0 goes to MEM0, Vref+, VeRef-
     ADC12MCTL1 = EOS + INCH_1 + SREF_5 ;     // A1 goes to MEM1,Vref+, VeRef-
    ADC12IE = 0x02;                           // Enable ADC12IFG.1 for ADC12MEM1
    ADC12CTL0 |= ENC;                         // Enable conversions
     _EINT();                                                 // Enable global Interrupts
} //init



void InitLCD(void)
{
	int i;
	for(i = 0; i < LCDMEMS; ++i) {		// Clear LCD memories used
		LCDMem[i] = 0;
	}
	P5SEL = BIT4|BIT3|BIT2;				// Select COM[3:1] function
	LCDAPCTL0 = LCDS4|LCDS8|LCDS12|LCDS16|LCDS20|LCDS24;	
						// Enable LCD segs 4-27 (4-25 used)
	LCDAVCTL0 = 0;				// No charge pump, everything internal
	LCDACTL = LCDFREQ_128 | LCD4MUX | LCDSON | LCDON;	
						// ACLK/128, 4mux, segments on, LCD_A on
}


// Interrupt service routine for port 1 inputs
// Only one bit is active so no need to check which
// clear any pending interrupts
// Device returns to low power mode automatically after ISR
// ----------------------------------------------------------------------
#pragma vector = PORT1_VECTOR
__interrupt void PORT1_ISR (void)
{

  DisplayReady();
  
  do {
    P1IFG = 0;                        // Clear any pending interrupts ...
      } while (P1IFG != 0);           // ... until none remain
   BTCTL &= ~BIT6;
   BTCNT1 = 0x00;
   BTCNT2 = 0x00;
   BTCTL = BTDIV + BTIP0 + BTIP1 + BTIP2;           // ACLK/(256*256) one second delay
  IE2 |= BTIE;                                // Enable BT interrupt
}

// Interrupt service routine for port ADC12
// After conversion results are stored in data array
// Turn back to low power mode
// ----------------------------------------------------------------------
#pragma vector = ADC12_VECTOR                                  // ADC12 ISR
__interrupt void ADC12ISR (void)
{                                      
 
    
  A0results[Index] = ADC12MEM0;             // Move A0 results, IFG is cleared
  A1results[Index] = ADC12MEM1;             // Move A1 results, IFG is cleared
  Index = (Index + 1);                // Increment results index, modulo
 
  __no_operation();                         // breakpoint to see ADC results                                         
   __bic_SR_register_on_exit(LPM3_bits);  // Exit LPM3 on return
  
}// ADC12ISR


// Interrupt service routine for Basic timer
// After button press Basic timer provides 1 sec delay for displaying Ready
// Then in ISR , timerA is initialized to start conversion
// ----------------------------------------------------------------------
// Basic Timer Interrupt Service Routine
#pragma vector=BASICTIMER_VECTOR
__interrupt void basic_timer_ISR(void)
{
  DisplayGo();
  IE2 &= ~BIT7; 
    TACTL = TASSEL_1 + MC_1 + TACLR;                          // ACLK, Clear TAR, Up Mode
    TACCTL1 = OUTMOD_2;                                      // Set / Reset
    TACCR0 = 255;                                             // 128 samples per second
    TACCR1 = 15;                                             //  

}


int filterbp2(int sample)                                     // Band Pass FIR filter for Beta Rhtym
{   static int buflp[32];                                    // Reserve 32 loactions for circular buffering
    static int offsetlp = 0;
    long z;
    int i;
    buflp[offsetlp] = sample;
    z = mul16(coeffsbp2[13], buflp[(offsetlp - 13) & 0x1F]);        //multiplication
    for (i = 0;  i < 13;  i++)
    z += mul16(coeffsbp2[i], buflp[(offsetlp - i) & 0x1F] + buflp[(offsetlp - 26 + i) & 0x1F]);
    offsetlp = (offsetlp + 1) & 0x1F;
    return  z >> 15;                                         // Return filter output
}


int filterbp1(int sample)                                     //  Band Pass FIR filter for mu Rhtym
{   static int buflp[32];                                    // Reserve 32 loactions for circular buffering
    static int offsetlp = 0;
    long z;
    int i;
    buflp[offsetlp] = sample;
    z = mul16(coeffsbp1[11], buflp[(offsetlp - 11) & 0x1F]);
    for (i = 0;  i < 11;  i++)
    z += mul16(coeffsbp1[i], buflp[(offsetlp - i) & 0x1F] + buflp[(offsetlp - 22 + i) & 0x1F]);
    offsetlp = (offsetlp + 1) & 0x1F;
    return  z >> 15;                                         // Return filter output
}



void calcNet(void)                       //Neural Network Calculation function
{
    //calculate the outputs of the hidden neurons
    //the hidden neurons are tanh
    int i = 0;
    for(i = 0;i<numHidden;i++)
    {
	  hiddenVal[i] = 0.0;

        for(int j = 0;j<numInputs;j++)
        {
	   hiddenVal[i] = hiddenVal[i] + (trainInputs[j]  * weightsIH[j][i]);
        }

        hiddenVal[i] = tanh(hiddenVal[i]);
    }

   //calculate the output of the network
   //the output neuron is linear
   outPred = 0.0;

   for(i = 0;i<numHidden;i++)
   {
    outPred = outPred + hiddenVal[i] * weightsHO[i];
   }
    //calculate the error
   // errThisPat = outPred - trainOutput[patNum];   // for training

}

void initWeights(void)                    // Neural network weights
{
  
weightsIH[0][0] = -8.951717;
weightsIH[1][0] = 7.653355;
weightsIH[2][0] = 6.631219;
weightsIH[3][0] = 11.094660;
weightsIH[0][1] = 3.310763;
weightsIH[1][1] = 3.215580;
weightsIH[2][1] = -0.963815;
weightsIH[3][1] = 6.746135;
weightsIH[0][2] = 3.641091;
weightsIH[1][2] = -1.118857;
weightsIH[2][2] = -2.280876;
weightsIH[3][2] = -12.060060;
weightsIH[0][3] = 11.269429;
weightsIH[1][3] = 6.643443;
weightsIH[2][3] = -6.614878;
weightsIH[3][3] = 14.733462;
weightsIH[0][4] = 0.743146;
weightsIH[1][4] = 10.348796;
weightsIH[2][4] = -15.558205;
weightsIH[3][4] = 18.596556;
weightsIH[0][5] = -12.275212;
weightsIH[1][5] = -1.859199;
weightsIH[2][5] = -20.228093;
weightsIH[3][5] = -12.789903;
weightsIH[0][6] = -21.430971;
weightsIH[1][6] = -3.899026;
weightsIH[2][6] = -12.496226;
weightsIH[3][6] = 4.471426;
weightsIH[0][7] = 1.179519;
weightsIH[1][7] = 8.862174;
weightsIH[2][7] = 7.262130;
weightsIH[3][7] = 0.380516;
weightsIH[0][8] = -3.315827;
weightsIH[1][8] = 0.415380;
weightsIH[2][8] = -0.804302;
weightsIH[3][8] = -7.525758;
weightsIH[0][9] = 8.997232;
weightsIH[1][9] = 6.043105;
weightsIH[2][9] = 4.525014;
weightsIH[3][9] = -4.655872;
weightsIH[0][10] = -2.567476;
weightsIH[1][10] = -1.001431;
weightsIH[2][10] = -5.790394;
weightsIH[3][10] = -12.538476;
weightsIH[0][11] = 0.964474;
weightsIH[1][11] = -6.690062;
weightsIH[2][11] = 5.976368;
weightsIH[3][11] = -26.924518;
weightsIH[0][12] = 6.004541;
weightsIH[1][12] = 5.625768;
weightsIH[2][12] = 10.801397;
weightsIH[3][12] = -0.778696;
weightsIH[0][13] = -2.619037;
weightsIH[1][13] = -11.472975;
weightsIH[2][13] = -9.598161;
weightsIH[3][13] = 0.805282;
weightsIH[0][14] = 2.341189;
weightsIH[1][14] = -7.747893;
weightsIH[2][14] = 0.036407;
weightsIH[3][14] = -11.517146;
weightsIH[0][15] = 2.618314;
weightsIH[1][15] = 2.424869;
weightsIH[2][15] = -7.886291;
weightsIH[3][15] = -12.294660;
weightsIH[0][16] = -1.006894;
weightsIH[1][16] = -3.148300;
weightsIH[2][16] = -10.965334;
weightsIH[3][16] = -2.622260;
weightsIH[0][17] = 1.943215;
weightsIH[1][17] = 9.396184;
weightsIH[2][17] = 8.086076;
weightsIH[3][17] = 0.148540;
weightsIH[0][18] = -7.163011;
weightsIH[1][18] = -5.225458;
weightsIH[2][18] = 15.685931;
weightsIH[3][18] = -16.417650;
weightsIH[0][19] = -11.894525;
weightsIH[1][19] = 18.598566;
weightsIH[2][19] = -1.046420;
weightsIH[3][19] = 14.616560;
weightsHO[0] = -1.369972;
weightsHO[1] = 0.576864;
weightsHO[2] = -0.742043;
weightsHO[3] = -0.751341;
weightsHO[4] = 0.608129;
weightsHO[5] = 1.522254;
weightsHO[6] = -1.743967;
weightsHO[7] = 0.138998;
weightsHO[8] = -0.989855;
weightsHO[9] = -2.040964;
weightsHO[10] = 1.837575;
weightsHO[11] = 1.636948;
weightsHO[12] = 1.512855;
weightsHO[13] = 0.959156;
weightsHO[14] = -1.104560;
weightsHO[15] = -1.112830;
weightsHO[16] = -1.348077;
weightsHO[17] = -0.206748;
weightsHO[18] = -0.431466;
weightsHO[19] = 1.069858;

}


void DisplayHello(void){
        LCDMem[1] = LCDHexChar[9];                    //Display hello on LCD
	LCDMem[2] = LCDHexChar[3];
	LCDMem[3] = LCDHexChar[3];
        LCDMem[4] = LCDHexChar[14];
        LCDMem[5] = LCDHexChar[2];
            }
void DisplayResult1(void){
      LCDMem[1] = LCDHexChar[3];                   //Display Clas: sol on LCd
	LCDMem[2] = LCDHexChar[9];
	LCDMem[3] = LCDHexChar[11];
         LCDMem[4] = LCDBlankChar | SEG_H;
         LCDMem[5] = LCDHexChar[15];
            }
            
void DisplayResult2(void){
  LCDMem[1] = LCDHexChar[12];                   //Display Clas: sag on LCD
	LCDMem[2] = LCDHexChar[10];
	LCDMem[3] = LCDHexChar[11];
         LCDMem[4] = LCDBlankChar | SEG_H;
         LCDMem[5] = LCDHexChar[15];
         
            }
void DisplayReady(void){
        LCDMem[1] = LCDHexChar[6];                   //Display ready on LCD
	LCDMem[2] = LCDHexChar[13];
	LCDMem[3] = LCDHexChar[10];
        LCDMem[4] = LCDHexChar[14];
        LCDMem[5] = LCDHexChar[5];
            }
            
void DisplayGo(void){
        LCDMem[1] = LCDHexChar[7];                   //Display start on LCD
	LCDMem[2] = LCDHexChar[5];
	LCDMem[3] = LCDHexChar[10];
        LCDMem[4] = LCDHexChar[7];
        LCDMem[5] = LCDHexChar[11];
            }
            
void DisplayStop(void){
  
        LCDMem[1] = LCDHexChar[4];                   //Display stop on LCD
	LCDMem[2] = LCDHexChar[9];
	LCDMem[3] = LCDHexChar[7];
        LCDMem[4] = LCDHexChar[11];
        LCDMem[5] = LCDBlankChar;
            }
//end of main.c
