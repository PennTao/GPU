varying vec2 v_Texcoords;
uniform sampler2D u_image;
uniform vec2 u_step;
const vec3 W = vec3(0.2125, 0.7154, 0.0721);

const int KERNEL_WIDTH = 3; // Odd
const float offset = 1.0;

void main(void)
{


	

	float luminance[9];

	for(int i=0; i<KERNEL_WIDTH;++i)
	{
		for (int j = 0; j < KERNEL_WIDTH; ++j)
		{
			vec2 coord = vec2(v_Texcoords.s + ((float(i) - offset) * u_step.s), v_Texcoords.t + ((float(j) - offset) * u_step.t));
			vec3 outp = texture2D(u_image, coord).rgb;
	        luminance[3*i+j] = dot(outp ,W);
		}
	}

	float weight_x[9];
	weight_x[0]= -1.0;
	weight_x[1]= -2.0;           
	weight_x[2]= -1.0;
	weight_x[3]= 0.0;
	weight_x[4]= 0.0;
	weight_x[5]= 0.0;
	weight_x[6]= 1.0;
	weight_x[7]= 2.0;
	weight_x[8]= 1.0;

	float weight_y[9];


	weight_y[0]= -1.0;
	weight_y[1]= 0.0;           
	weight_y[2]= 1.0;
	weight_y[3]= -2.0;
	weight_y[4]= 0.0;
	weight_y[5]= 2.0;
	weight_y[6]= -1.0;
	weight_y[7]= 0.0;
	weight_y[8]= 1.0;

	float x=0.0,y=0.0;
	for(int i=0;i<9;i++)
	{
		x+=weight_x[i]*luminance[i];
		y+=weight_y[i]*luminance[i];
	}

	float m=sqrt(x*x+y*y);

	vec3 r=texture2D(u_image, v_Texcoords).rgb;
	if(m<0.7)
	{
		float quantize = 0.9; // determine it
		r *= quantize;
		r += vec3(0.5);
		ivec3 irgb = ivec3(r);
		r = vec3(irgb) / quantize;
	}
	else
	{
		r=vec3(0.0,0.0,0.0);
	}
    gl_FragColor = vec4(r, 1.0);


}
