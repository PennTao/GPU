/*varying vec2 v_Texcoords;

uniform sampler2D u_image;
uniform vec2 u_step;

const int KERNEL_WIDTH = 3; // Odd
const float offset = 0.0;

void main(void)
{
    vec3 accum = vec3(0.0);

	for (int i = 0; i < KERNEL_WIDTH; ++i)
	{
		for (int j = 0; j < KERNEL_WIDTH; ++j)
		{
			vec2 coord = vec2(v_Texcoords.s + ((float(i) - offset) * u_step.s), v_Texcoords.t + ((float(j) - offset) * u_step.t));
			//accum +=float(((2-max(i-1,1-i))*(2-max(j-1,1-j))))* texture2D(u_image, coord).rgb;
			accum = texture2D(u_image,coord).rgb;
		}
	}	
	//accum = accum/16.0;
    gl_FragColor = vec4(accum / float(KERNEL_WIDTH * KERNEL_WIDTH), 1.0);
}
*/
varying vec2 v_Texcoords;

uniform sampler2D u_image;
uniform vec2 u_step;

const int KERNEL_WIDTH = 3; // Odd
const float offset = 1.0;

void main(void)
{
    vec3 accum = vec3(0.0);

	for (int i = 0; i < KERNEL_WIDTH; ++i)
	{
		for (int j = 0; j < KERNEL_WIDTH; ++j)
		{
			vec2 coord = vec2(v_Texcoords.s + ((float(i) - offset) * u_step.s), v_Texcoords.t + ((float(j) - offset) * u_step.t));
			accum += ((2.0-abs(float(i)-1.0))*(2.0-abs(float(j)-1.0)))*texture2D(u_image, coord).rgb;
		}
	}	

    gl_FragColor = vec4(accum / 16.0, 1.0);
}
