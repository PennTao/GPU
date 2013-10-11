#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <GL/glew.h>
#include <GL/glut.h>
#include <iostream>

#include "Utility.h"
#include "TypedArray.h"

static const int NUM_WIDTH_PTS = 32;
static const int NUM_HEIGHT_PTS = 32;

Float32Array *heights;
static unsigned int numberOfIndices;

GLuint positionLocation = 0;
GLuint heightLocation = 1;

GLint u_modelViewPerspectiveLocation;


//Tao
GLint location_uniform;
GLfloat delta = 0.01;
GLfloat time_update = 0;
void initShader() {
	GLuint program = Utility::createProgram("vs.glsl", "fs.glsl");
	//Tao
	location_uniform = glGetUniformLocation(program,"u_time");
	//
	glBindAttribLocation(program, positionLocation, "position");
    u_modelViewPerspectiveLocation = glGetUniformLocation(program,"u_modelViewPerspective");
	

	glUseProgram(program);
}

void uploadMesh(const Float32Array &positions, const Float32Array *heights, const Uint16Array& indices)
{
    // Positions
    GLuint positionsName;
	glGenBuffers(1,&positionsName);
	glBindBuffer(GL_ARRAY_BUFFER, positionsName);
	glBufferData(GL_ARRAY_BUFFER, positions.length * positions.BYTES_PER_ELEMENT, positions, GL_STATIC_DRAW);
	glVertexAttribPointer(positionLocation, 2, GL_FLOAT, GL_FALSE,0,0);
    glEnableVertexAttribArray(positionLocation);

    if (heights)
    {
        // Heights
        GLuint heightsName;
	    glGenBuffers(1,&heightsName);
	    glBindBuffer(GL_ARRAY_BUFFER, heightsName);
	    glBufferData(GL_ARRAY_BUFFER, heights->length * heights->BYTES_PER_ELEMENT, 0, GL_STREAM_DRAW);
	    glVertexAttribPointer(heightLocation, 1, GL_FLOAT, GL_FALSE,0,0);
        glEnableVertexAttribArray(heightLocation);
    }

    // Indices
    GLuint indicesName;
	glGenBuffers(1,&indicesName);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indicesName);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.length * indices.BYTES_PER_ELEMENT, indices, GL_STATIC_DRAW);
}

void initGrid()
{
    static const int WIDTH_DIVISIONS = NUM_WIDTH_PTS - 1;
    static const int HEIGHT_DIVISIONS = NUM_HEIGHT_PTS - 1;

    const int numberOfPositions = NUM_WIDTH_PTS * NUM_HEIGHT_PTS;

    Float32Array positions(2 * numberOfPositions);
    Uint16Array indices(2 * ((NUM_HEIGHT_PTS * (NUM_WIDTH_PTS - 1)) + (NUM_WIDTH_PTS * (NUM_HEIGHT_PTS - 1))));

    int positionsIndex = 0;
    int indicesIndex = 0;

	for (int j = 0; j < NUM_WIDTH_PTS; ++j) 
    {
        positions[positionsIndex++] = (float)(j)/(NUM_WIDTH_PTS - 1);
        positions[positionsIndex++] = 0.0f;

		if (j>=1) 
        {
			unsigned short length = (unsigned short)(positionsIndex / 2);
            indices[indicesIndex++] = length - 2;
            indices[indicesIndex++] = length - 1;
		}
	}

	for (int i = 0; i < HEIGHT_DIVISIONS; ++i) 
    {
         float v = (float)(i + 1)/(NUM_HEIGHT_PTS - 1);
         positions[positionsIndex++] = 0.0f;
         positions[positionsIndex++] = v;

         unsigned short length = (unsigned short)(positionsIndex / 2);
         indices[indicesIndex++] = length - 1;
         indices[indicesIndex++] = length - 1 - NUM_WIDTH_PTS;

		 for (int j = 0; j < WIDTH_DIVISIONS; j++) 
         {
             positions[positionsIndex++] = (float)(j + 1)/(NUM_WIDTH_PTS - 1);
             positions[positionsIndex++] = v;

			 unsigned short length = (unsigned short)(positionsIndex / 2);
			 unsigned short new_pt = length - 1;
             indices[indicesIndex++] = new_pt - 1;  // Previous side
             indices[indicesIndex++] = new_pt;

             indices[indicesIndex++] = new_pt - NUM_WIDTH_PTS;  // Previous bottom
             indices[indicesIndex++] = new_pt;
		 }
	}

    uploadMesh(positions, heights, indices);
    numberOfIndices = indices.length;
}

const float ROTATION_STEP_STEP = 0.01f;
float rotation_step = 0.03f;

void adjustRotation(float adjustment) {
    rotation_step += adjustment;
}

void speedUpRotation() {
    adjustRotation(ROTATION_STEP_STEP);
}

void slowDownRotation() {
    adjustRotation(-ROTATION_STEP_STEP);
}

static float object_rotation;

glm::mat4 update_rotation() {
	object_rotation += rotation_step;
	if (object_rotation >= 360.0f) object_rotation = 0.0f;
    glm::vec3 axis(0.0f,0.0f,1.0f);
	return glm::rotate(glm::mat4(), object_rotation, axis);
}

static glm::mat4 view;
static glm::mat4 persp;

void display(void)
{
    ///////////////////////////////////////////////////////////////////////////
    // Update
	//Tao
	time_update = time_update + delta;
	//
	glm::mat4 model = glm::translate(update_rotation(), glm::vec3(-0.5, -0.5, 0.0));
    glm::mat4 mvp = persp * view * model;
	//Tao
	glUniform1f(location_uniform,time_update);
    ///////////////////////////////////////////////////////////////////////////
    // Render

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Shader and vertex buffers already bound
	glUniformMatrix4fv(u_modelViewPerspectiveLocation,1,GL_FALSE,&mvp[0][0]);
	glDrawElements(GL_LINES, numberOfIndices, GL_UNSIGNED_SHORT,0);

    glutPostRedisplay();
	glutSwapBuffers();
}

void reshape(int w, int h)
{
	glViewport(0,0,(GLsizei)w,(GLsizei)h);
    persp = glm::perspective(45.0f, 0.5f, 0.1f, 100.0f);
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
	   case '+':
           speedUpRotation();
		   break;
	   case '-':
           slowDownRotation();
		   break;	
	}
}

void init()
{
    glm::vec3 eye(2.0f,1.0f,3.0f);
	glm::vec3 center(0.0f,0.0f,0.0f);
    glm::vec3 up(0.0f,0.0f,1.0f);
    view = glm::lookAt(eye,center,up);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(1027, 768);
	glutCreateWindow("Meshes with Shaders");
	glewInit();
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		std::cout << "glewInit failed, aborting." << std::endl;
		exit (1);
	}
	std::cout << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;
	std::cout << "OpenGL version " << glGetString(GL_VERSION) << " supported" << std::endl;
    
    init();
	initGrid();
    initShader();
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);	
    glutKeyboardFunc(keyboard);
	
	
	glutMainLoop();
	return 0;
}