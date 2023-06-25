// Rami-Pastrami 2023
// This Shader include file has some commonly used functions for HLSL neural networks

//////////////////////////////////////////////////////////////////////
/////////////////////////// Loading Stuff ////////////////////////////
//////////////////////////////////////////////////////////////////////

// Used to load weights / biases from PNG textures into the CRT as floats
// TODO perhaps we should read the bits directly instead and translate to float notation? This is accurate enough for now though and not a bottleneck
float TexPixelToFloat(float4 colorIn, float normalizer) 
{
	int RGBAint = (floor(colorIn[0] * 256.0) * 1000000) + (floor(colorIn[1] * 256.0) * 10000) + (floor(colorIn[2] * 256.0) * 100) + (floor(colorIn[3] * 256.0));
	return (((float)RGBAint / 100000000.0) - 1.0) * normalizer;
}

// Generic function to read a weight value from a Linear CRT data holder
float GetWeightValFromLinearData(int numNeuronsIn, int numNeuronsOut, int curInputNeuron, int curOutputNeuronDepth, sampler2D dataTex)
{
	float h = (float)curOutputNeuronDepth / (float)numNeuronsOut;
	float w = (float)curInputNeuron / ((float)numNeuronsIn + 1.0);
	return tex2D(dataTex, float2(w,h));
}

// Generic function to read a bias value from a Linear CRT data holder
float GetBiasValFromLinearData(int numNeuronsIn, int numNeuronsOut, int curOutputNeuronDepth, sampler2D dataTex)
{
	float h = (float)curOutputNeuronDepth / (float)numNeuronsOut;
	float w = (float)numNeuronsIn / ((float)numNeuronsIn + 1.0);
	return tex2D(dataTex, float2(w,h));
}

//////////////////////////////////////////////////////////////////////
/////////////////////// Activation Functions /////////////////////////
//////////////////////////////////////////////////////////////////////

float Activation_Tanh(float input)
{
	return tanh(input);  // duh
}

