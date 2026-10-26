#version 450
//============================================================================================================
//
//
//                  Copyright (c) 2025, Qualcomm Innovation Center, Inc. All rights reserved.
//                              SPDX-License-Identifier: BSD-3-Clause
//
//============================================================================================================
// Snapdragon Game Super Resolution 1 at the present (vk_fsr.c, TOTK_UPSCALE=sgsr; builtin/sgsr/LICENSE): Qualcomm's
// sgsr1_shader_mobile_edge_direction.frag (github.com/SnapdragonStudios/snapdragon-gsr, sgsr/v1/include/glsl) in RGBA mode
// with the edge direction on, drawn straight into the window. Changed: its uniforms are push constants, and the texture
// coordinate comes from the window position (Map) instead of a vertex output.

precision mediump float;
precision highp int;

#define OperationMode 1
#define UseEdgeDirection
#define EdgeThreshold 8.0/255.0
#define EdgeSharpness 2.0

layout(push_constant) uniform P { highp vec4 ViewportInfo; highp vec4 Map; } pc;   // ViewportInfo: 1/w, 1/h, w, h of the texture; Map: window position to texture coordinate (scale xy, offset zw)
layout(set = 0, binding = 0) uniform mediump sampler2D ps0;
layout(location = 0) out vec4 out_Target0;

float fastLanczos2(float x)
{
	float wA = x-4.0;
	float wB = x*wA-wA;
	wA *= wA;
	return wB*wA;
}

#if defined(UseEdgeDirection)
vec2 weightY(float dx, float dy, float c, vec3 data)
#else
vec2 weightY(float dx, float dy, float c, float data)
#endif
{
#if defined(UseEdgeDirection)
	float std = data.x;
	vec2 dir = data.yz;

	float edgeDis = ((dx*dir.y)+(dy*dir.x));
	float x = (((dx*dx)+(dy*dy))+((edgeDis*edgeDis)*((clamp(((c*c)*std),0.0,1.0)*0.7)+-1.0)));
#else
	float std = data;
	float x = ((dx*dx)+(dy* dy))* 0.55 + clamp(abs(c)*std, 0.0, 1.0);
#endif

	float w = fastLanczos2(x);
	return vec2(w, w * c);	
}

vec2 edgeDirection(vec4 left, vec4 right)
{
	vec2 dir;
	float RxLz = (right.x + (-left.z));
	float RwLy = (right.w + (-left.y));
	vec2 delta;
	delta.x = (RxLz + RwLy);
	delta.y = (RxLz + (-RwLy));
	float lengthInv = inversesqrt((delta.x * delta.x+ 3.075740e-05) + (delta.y * delta.y));
	dir.x = (delta.x * lengthInv);
	dir.y = (delta.y * lengthInv);
	return dir;
}

void main()
{
	highp vec2 tc = gl_FragCoord.xy * pc.Map.xy + pc.Map.zw;   // the texture coordinate of this pixel of the window
	vec4 color;
	if(OperationMode == 1)
		color.xyz = textureLod(ps0,tc,0.0).xyz;
	else
		color.xyzw = textureLod(ps0,tc,0.0).xyzw;

	highp float xCenter;
	xCenter = abs(tc.x+-0.5);
	highp float yCenter;
	yCenter = abs(tc.y+-0.5);

	//todo: config the SR region based on needs
	//if ( OperationMode!=4 && xCenter*xCenter+yCenter*yCenter<=0.4 * 0.4)
	if ( OperationMode!=4)
	{
		highp vec2 imgCoord = ((tc*pc.ViewportInfo.zw)+vec2(-0.5,0.5));
		highp vec2 imgCoordPixel = floor(imgCoord);
		highp vec2 coord = (imgCoordPixel*pc.ViewportInfo.xy);
		vec2 pl = (imgCoord+(-imgCoordPixel));
		vec4  left = textureGather(ps0,coord, OperationMode);

		float edgeVote = abs(left.z - left.y) + abs(color[OperationMode] - left.y)  + abs(color[OperationMode] - left.z) ;
		if(edgeVote > EdgeThreshold)
		{
			coord.x += pc.ViewportInfo.x;

			vec4 right = textureGather(ps0,coord + vec2(pc.ViewportInfo.x, 0.0), OperationMode);
			vec4 upDown;
			upDown.xy = textureGather(ps0,coord + vec2(0.0, -pc.ViewportInfo.y),OperationMode).wz;
			upDown.zw  = textureGather(ps0,coord+ vec2(0.0, pc.ViewportInfo.y), OperationMode).yx;

			float mean = (left.y+left.z+right.x+right.w)*0.25;
			left = left - vec4(mean);
			right = right - vec4(mean);
			upDown = upDown - vec4(mean);
			color.w =color[OperationMode] - mean;

			float sum = (((((abs(left.x)+abs(left.y))+abs(left.z))+abs(left.w))+(((abs(right.x)+abs(right.y))+abs(right.z))+abs(right.w)))+(((abs(upDown.x)+abs(upDown.y))+abs(upDown.z))+abs(upDown.w)));				
			float sumMean = 1.014185e+01/sum;
			float std = (sumMean*sumMean);	

#if defined(UseEdgeDirection)
			vec3 data = vec3(std, edgeDirection(left, right));
#else
			float data = std;
#endif
			vec2 aWY = weightY(pl.x, pl.y+1.0, upDown.x,data);				
			aWY += weightY(pl.x-1.0, pl.y+1.0, upDown.y,data);
			aWY += weightY(pl.x-1.0, pl.y-2.0, upDown.z,data);
			aWY += weightY(pl.x, pl.y-2.0, upDown.w,data);			
			aWY += weightY(pl.x+1.0, pl.y-1.0, left.x,data);
			aWY += weightY(pl.x, pl.y-1.0, left.y,data);
			aWY += weightY(pl.x, pl.y, left.z,data);
			aWY += weightY(pl.x+1.0, pl.y, left.w,data);
			aWY += weightY(pl.x-1.0, pl.y-1.0, right.x,data);
			aWY += weightY(pl.x-2.0, pl.y-1.0, right.y,data);
			aWY += weightY(pl.x-2.0, pl.y, right.z,data);
			aWY += weightY(pl.x-1.0, pl.y, right.w,data);

			float finalY = aWY.y/aWY.x;
			float maxY = max(max(left.y,left.z),max(right.x,right.w));
			float minY = min(min(left.y,left.z),min(right.x,right.w));
			float deltaY = clamp(EdgeSharpness*finalY, minY, maxY) -color.w;			
			
			//smooth high contrast input
			deltaY = clamp(deltaY, -23.0 / 255.0, 23.0 / 255.0);

			color.x = clamp((color.x+deltaY),0.0,1.0);
			color.y = clamp((color.y+deltaY),0.0,1.0);
			color.z = clamp((color.z+deltaY),0.0,1.0);
		}
	}

	color.w = 1.0;  //assume alpha channel is not used
	out_Target0.xyzw = color;
}