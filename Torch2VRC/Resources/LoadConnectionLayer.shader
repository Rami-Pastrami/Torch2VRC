Shader "Rami-Pastrami/Torch2VRC/LoadConnectionLayer"
{
    Properties
    {
        _Normalization_Weights("Weight Normalization", float) = 1.0
        _Normalization_Biases("Bias Normalization", float) = 1.0

        _TexWeights("Weights", 2D) = "black" {}
        _TexBias("Bias", 2D) = "black" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Cull Off
        Lighting Off
        ZWrite Off
        ZTest Always

        Pass
        {
            CGINCLUDE //The following will be included in all passes of the shader
            #include "UnityCustomRenderTexture.cginc"
            #include "NN_Common.cginc"
            #pragma vertex CustomRenderTextureVertexShader //CRT vertex
            #pragma fragment frag noshadow
            #include "UnityCG.cginc"
            ENDCG


            Name "Load Weights"
            CGPROGRAM

            uniform float _Normalization_Weights;
            uniform sampler2D _TexWeights;

            float frag (v2f_customrendertexture i) : SV_Target
            {
                // Sample and load in texture
                float4 col = tex2D(_TexWeights, i.localTexcoord);
                return TexPixelToFloat(col, _Normalization_Weights );
            }

            ENDCG
        }
        
        Pass
        {
            Name "Load Bias"
            CGPROGRAM

            uniform float _Normalization_Biases;
            uniform sampler2D _TexBias;

            float frag (v2f_customrendertexture i) : SV_Target
            {
                // Sample and load in texture
                float4 col = tex2D(_TexBias, i.localTexcoord.yx);
                return TexPixelToFloat(col, _Normalization_Biases );
            }

            ENDCG


        }

    }
}
