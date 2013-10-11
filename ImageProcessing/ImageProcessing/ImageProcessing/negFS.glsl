varying vec2 v_Texcoords;

uniform sampler2D u_image;

void main(void)
{
	vec3 outp;
	outp = vec3(0.0);
	outp = texture2D(u_image, v_Texcoords).rgb;
	outp = vec3(1.0) - outp;
	gl_FragColor = vec4(outp,1);
//	gl_FragColor = texture2D(u_image, v_Texcoords);
}
