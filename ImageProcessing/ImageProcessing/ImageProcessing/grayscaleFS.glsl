varying vec2 v_Texcoords;

uniform sampler2D u_image;
const vec3 W = vec3(0.2125, 0.7154, 0.0721);
void main(void)
{
	vec3 outp =vec3(0.0);
	outp = texture2D(u_image, v_Texcoords).rgb;
	float luminance = dot(outp ,W);
	outp.rgb = vec3(luminance);
	gl_FragColor = vec4(outp,1.0);
}
