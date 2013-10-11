attribute vec2 position;

uniform mat4 u_modelViewPerspective;
uniform float u_time;
varying vec3 color;
void main(void)
{


//	u_time = u_time + 11;
//	if(u_time>3.14)
//	  u_time=0;
	float s_contrib = sin(position.x*2.0*3.14159 + u_time);
	float t_contrib = cos(position.y*2.0*3.14159 + u_time);
	float height = s_contrib*t_contrib;
//    float height = 0.0;
	color = vec3((height+1)/2, 0.5,0.5);
    gl_Position = u_modelViewPerspective * vec4(vec3(position, height), 1.0);

}