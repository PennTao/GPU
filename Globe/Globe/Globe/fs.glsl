//View-Space directional light
//A unit vector
#define CLOUD_TH 0.7    // 
#define BUMP_MAGNIFY 1 
#define CLOUD_MIX   1   //cloud mix ratio <=1
uniform vec3 u_CameraSpaceDirLight;

//Diffuse texture map for the day
uniform sampler2D u_DayDiffuse;
//Ambient texture map for the night side
uniform sampler2D u_Night;
//Color map for the clouds
uniform sampler2D u_Cloud;
//Transparency map for the clouds.  Note that light areas are where clouds are NOT
//Dark areas are were clouds are present
uniform sampler2D u_CloudTrans;
//Mask of which areas of the earth have specularity
//Oceans are specular, landmasses are not
uniform sampler2D u_EarthSpec;
//Bump map
uniform sampler2D u_Bump;

uniform float u_time;
uniform mat4 u_InvTrans;

varying vec3 v_Normal;				// surface normal in camera coordinates
varying vec2 v_Texcoord;
varying vec3 v_Position;			// position in camera coordinates
varying vec3 v_positionMC;			// position in model coordinates

mat3 eastNorthUpToEyeCoordinates(vec3 positionMC, vec3 normalEC);
vec4 base = vec4(0,0,0,0);

void main(void)
{
    vec3 normal = normalize(v_Normal);            // surface normal - normalized after rasterization
	vec3 eyeToPosition = normalize(v_Position);   // normalized eye-to-position vector in camera coordinates
	
	
	vec4 bump = texture2D(u_Bump,v_Texcoord);
	vec4 right = texture2D(u_Bump,vec2(v_Texcoord.x + BUMP_MAGNIFY * 0.001,v_Texcoord.y));
	vec4 top = texture2D(u_Bump,vec2(v_Texcoord.x,v_Texcoord.y -  BUMP_MAGNIFY * 0.002));
	vec3 bumpNormal = normalize(vec3((bump - right).x, (bump - top).y,0.2));
	//transfer to eyecoord
	bumpNormal = eastNorthUpToEyeCoordinates(v_positionMC,bumpNormal)*bumpNormal ;
//	bumpNormal = bumpNormal * eastNorthUpToEyeCoordinates(v_positionMC,bumpNormal) ;
    float diffuseLand = max(dot(u_CameraSpaceDirLight, bumpNormal), 0.0);
	float diffuseCloud =  max(dot(u_CameraSpaceDirLight, normal), 0.0);
	//float diffuse = dot(u_CameraSpaceDirLight, normal);
	vec3 toReflectedLight = reflect(-u_CameraSpaceDirLight, normal);
	float specular = max(dot(toReflectedLight, -eyeToPosition), 0.0);
	specular = pow(specular, 20.0);
	vec4 day = texture2D(u_DayDiffuse,v_Texcoord);
	vec4 landspec  = texture2D(u_EarthSpec,v_Texcoord);
	vec4 night = texture2D(u_Night,v_Texcoord);
	v_Texcoord.s = v_Texcoord.s + u_time;
	vec4 cloud = texture2D(u_Cloud,v_Texcoord);
	vec4 cloudtrans = texture2D(u_CloudTrans,v_Texcoord);
	
	if(diffuseCloud > 0)
	{	
		vec4 cloudlit;
		vec4 daylit;		
		//vec4 nightlit;
		if(landspec.r==landspec.g==landspec.b == 0)
		{
			daylit = (base+(0.8 * diffuseLand))*day;
			
		}
		else
		{
			daylit = (base + (0.8 * diffuseLand) + (0.4 * specular))*day;
		}
		//nightlit = (base+(0.8 * diffuseLand))*night;
	    cloudlit = (base+(0.8 * diffuseCloud))*cloud ;
			


		if(cloudtrans.r<CLOUD_TH && cloudtrans.g<CLOUD_TH && cloudtrans.b<CLOUD_TH)
		{
			
			vec4 daymix = mix(daylit,cloudlit,CLOUD_MIX);

			vec4 nightmix = mix(night,cloudtrans,CLOUD_MIX);
			gl_FragColor = mix(daymix,nightmix,1-diffuseCloud);
			//gl_FragColor = mix(day,night,1-diffuse);
		}	
		else
			gl_FragColor = mix(daylit,night,1-diffuseCloud);
	}
	else
	{
		
		//vec4 nightlit =  (base+(0.8 * diffuseLand))*night;
		vec4 nightmix = mix(night,cloudtrans,CLOUD_MIX);
		if(cloudtrans.r<CLOUD_TH && cloudtrans.g<CLOUD_TH && cloudtrans.b<CLOUD_TH)
			gl_FragColor = nightmix;
		//	gl_FragColor = night;
		else
			gl_FragColor = night;
	}
//	vec4 isLand = texture2D(u_EarthSpec,v_Texcoord);
//	vec4 isCloud = texture2D(u_CloudTrans,v_Texcoord);

/*	if(diffuse > 0)
	{
		if(isCloud.r <0.7 && isCloud.g < 0.7 && isCloud.b <0.7)
		{
		//	gl_FragColor = mix(( mix(texture2D(u_DayDiffuse,v_Texcoord),texture2D(u_Cloud,v_Texcoord),0.9)),mix(texture2D(u_Night,v_Texcoord),isCloud,0.9),1-diffuse);
			gl_FragColor = mix((mix(texture2D(u_DayDiffuse,v_Texcoord),texture2D(u_Cloud,v_Texcoord),0.9)),mix(texture2D(u_Night,v_Texcoord),isCloud,0.9),1-diffuse);

			 
		}
		else 
			gl_FragColor = mix(texture2D(u_DayDiffuse,v_Texcoord),mix(texture2D(u_Night,v_Texcoord),isCloud,0.9),1-diffuse);
		
	}
	else
	{
		if(isCloud.r <0.7 && isCloud.g < 0.7 && isCloud.b <0.7)
		{	
			gl_FragColor = mix(texture2D(u_Night,v_Texcoord),isCloud,0.9);
		}
		else
			gl_FragColor = texture2D(u_Night,v_Texcoord);
		
	}*/
	//gl_FragColor = texture2D(u_DayDiffuse,1.03*v_Texcoord);
//	if(diffuse > 0.5)
//	gl_FragColor = texture2D(u_DayDiffuse,v_Texcoord);
//		//gl_FragColor = mix(diffuse,specular,0.6) * texture2D(u_DayDiffuse, v_Texcoord);
//		gl_FragColor = ((0.8 * diffuse) + (0.4 * specular)) * texture2D(u_DayDiffuse, v_Texcoord);
/*    if(diffuse > 0)
	{
		vec4 isLand = texture2D(u_EarthSpec,v_Texcoord);
		vec4 isCloud = texture2D(u_CloudTrans,v_Texcoord);
		if((isCloud.r < 0.9)&&(isCloud.g<0.9)||(isCloud.b<0.9))
			gl_FragColor = 0.5*(base+(0.8 * diffuse)) * mix(mix(texture2D(u_DayDiffuse, v_Texcoord), texture2D(u_Night, v_Texcoord),1-diffuse),texture2D(u_Cloud,v_Texcoord),0.65);
		else 
			gl_FragColor = 0.5*(base+((0.8 * diffuse) + (0.4 * specular)))* mix(mix(texture2D(u_DayDiffuse, v_Texcoord), texture2D(u_Night, v_Texcoord),1-diffuse),texture2D(u_Cloud,v_Texcoord),0.05);
	}
	else
	{
		gl_FragColor =  0.5 * texture2D(u_Night,v_Texcoord);
	}*/
//	gl_FragColor = ((0.8 * diffuse) + (0.4 * specular)) * texture2D(u_Night, v_Texcoord);
}

mat3 eastNorthUpToEyeCoordinates(vec3 positionMC, vec3 normalEC)
{
    vec3 tangentMC = normalize(vec3(-positionMC.y, positionMC.x, 0.0));  // normalized surface tangent in model coordinates
    vec3 tangentEC = normalize(mat3(u_InvTrans) * tangentMC);            // normalized surface tangent in eye coordiantes
    vec3 bitangentEC = normalize(cross(normalEC, tangentEC));            // normalized surface bitangent in eye coordinates

    return mat3(
        tangentEC.x,   tangentEC.y,   tangentEC.z,
        bitangentEC.x, bitangentEC.y, bitangentEC.z,
        normalEC.x,    normalEC.y,    normalEC.z);
}