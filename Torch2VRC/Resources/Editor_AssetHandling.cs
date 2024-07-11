// Auto-Generated by Torch2VRC
// There should be no need to modify anything in this file, it will not be included in your VRC world!

using UnityEngine;
using UnityEditor;
using System.IO;
using UnityEditor.Experimental;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;


namespace Rami.Torch2VRC
{
    public static class WriteAsset
    {
        /// <summary>
        /// Creates a CustomRenderTexture Asset
        /// </summary>
        /// <param name="CRTPath">Unity path (including filename) of the CRT</param>
        /// <param name="materialPath">Unity path to the material</param>
        /// <param name="width">width of the CRT in pixels</param>
        /// <param name="height">height of the CRT in pixels</param>
        /// <param name="updateMode"></param>
        /// <param name="isDoubleBuffered"></param>
        /// <param name="updateZones">array of the update zones to use</param>
        public static void CustomRenderTexture(string CRTPath, string materialPath, int width, int height,
            CustomRenderTextureUpdateMode updateMode, bool isDoubleBuffered, CustomRenderTextureUpdateZone[] updateZones)
        {
            CustomRenderTexture crt = new CustomRenderTexture(width, height);
            crt.format = RenderTextureFormat.RHalf;
            crt.useMipMap = false;
            crt.wrapMode = TextureWrapMode.Clamp;
            crt.filterMode = FilterMode.Point;
            crt.anisoLevel = 0;
            crt.updateZoneSpace = CustomRenderTextureUpdateZoneSpace.Pixel;

            RenderTextureDescriptor descriptor = crt.descriptor;
            descriptor.depthBufferBits = 0;
            crt.descriptor = descriptor;

            PropertyInfo propertyInfo = typeof(CustomRenderTexture).GetProperty("enableCompatibleColorFormat", BindingFlags.Instance | BindingFlags.NonPublic);
            if (propertyInfo == null) { Debug.LogWarning("Unable to enable 'enableCompatibleColorFormat', please ensure it is enabled manually"); }
            else { propertyInfo.SetValue(crt, true); }

            Material layerMaterial = (Material)AssetDatabase.LoadAssetAtPath(materialPath, typeof(Material));
            crt.material = layerMaterial;
            crt.initializationMode = CustomRenderTextureUpdateMode.OnLoad;
            crt.initializationColor = Color.black;

            crt.updateMode = updateMode;
            crt.doubleBuffered = isDoubleBuffered;
            crt.SetUpdateZones(updateZones);

            AssetDatabase.CreateAsset(crt, CRTPath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

    }

    public static class ImportAsset
    {
        /// <summary>
        /// Imports a texture with proper settings for use in the neural network
        /// </summary>
        /// <param name="connectionTexturePath">Unity path to the image texture</param>
        public static void Texture(string connectionTexturePath)
        {
            TextureImporter textureImporter = AssetImporter.GetAtPath(connectionTexturePath) as TextureImporter;
            if (textureImporter == null) { Debug.LogError("Unable to load Texture file at " + connectionTexturePath); return; }

            TextureImporterPlatformSettings platformSettings = new TextureImporterPlatformSettings();
            textureImporter.textureType = TextureImporterType.Default;
            textureImporter.textureShape = TextureImporterShape.Texture2D;
            textureImporter.sRGBTexture = false;
            textureImporter.alphaSource = TextureImporterAlphaSource.FromInput;
            textureImporter.alphaIsTransparency = true;
            textureImporter.wrapMode = TextureWrapMode.Clamp;
            textureImporter.filterMode = FilterMode.Point;
            platformSettings.maxTextureSize = 8192;
            platformSettings.resizeAlgorithm = TextureResizeAlgorithm.Mitchell;
            platformSettings.format = TextureImporterFormat.RGBA32;
            textureImporter.SetPlatformTextureSettings(platformSettings);

            AssetDatabase.ImportAsset(connectionTexturePath);
            AssetDatabase.Refresh();
        }
    }
}