varying vec2 v_Texcoords;

uniform sampler2D u_image;
uniform vec2 u_step;

const int KERNEL_WIDTH = 3; // Odd
const float offset = 1.0;
const vec3 W = vec3(0.2125, 0.7154, 0.0721);
void main(void)
{
    vec3 sobelH = vec3(0.0);
	vec3 sobelV = vec3(0.0);
	vec3 grayS =vec3(0.0);
	vec3 outp;
	

	for (int i = 0; i < KERNEL_WIDTH; ++i)
	{
		for (int j = 0; j < KERNEL_WIDTH; ++j)
		{
			vec2 coord = vec2(v_Texcoords.s + ((float(i) - offset) * u_step.s), v_Texcoords.t + ((float(j) - offset) * u_step.t));
			grayS = texture2D(u_image, coord).rgb;
			float luminance = dot(grayS ,W);
			grayS.rgb = vec3(luminance);
			sobelH += ((float(i)- 1.0)*(2.0 - abs(float(j)-1.0)))*grayS;
		}
	}	
	for (int i = 0; i < KERNEL_WIDTH; ++i)
	{
		for (int j = 0; j < KERNEL_WIDTH; ++j)
		{
			vec2 coord = vec2(v_Texcoords.s + ((float(i) - offset) * u_step.s), v_Texcoords.t + ((float(j) - offset) * u_step.t));
			grayS = texture2D(u_image, coord).rgb;
			float luminance = dot(grayS ,W);
			grayS.rgb = vec3(luminance);
			sobelV +=((2.0 - abs(float(i)- 1.0))*(float(j)-1.0))*grayS;
		}
	}	
	outp.r = sqrt(sobelH.r * sobelH.r + sobelV.r * sobelV.r);
	
	outp.g = sqrt(sobelH.g * sobelH.g + sobelV.g * sobelV.g);
	
	outp.b = sqrt(sobelH.b * sobelH.b + sobelV.b * sobelV.b);
	outp = 3.0 * outp;
    gl_FragColor = vec4(outp / float(KERNEL_WIDTH * KERNEL_WIDTH), 1.0);
}
