#ifndef SNNDEMO_BOUNDINGBOXUTIL_H
#define SNNDEMO_BOUNDINGBOXUTIL_H

#include <glad/glad.h>

class BoundingBoxUtil {
private:
    const int widthDim = 416;
    const int heightDim = 416;

    const char *vertexShaderSource = "#version 320 es\n"
                                     "#define FLOAT_PRECISION mediump\n"
                                     "precision FLOAT_PRECISION float;\n"
                                     "layout (location = 0) in vec2 aPos;\n"
                                     "void main()\n"
                                     "{\n"
                                     "   gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);\n"
                                     "}\0";
    const char *fragmentShaderSource = "#version 320 es\n"
                                       "#define FLOAT_PRECISION mediump\n"
                                       "precision FLOAT_PRECISION float;\n"
                                       "out vec4 FragColor;\n"
                                       "void main()\n"
                                       "{\n"
                                       "   FragColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);\n"
                                       "}\n\0";
    gl::SimpleGlslProgram _rectangleProgram = gl::SimpleGlslProgram("rectangleProgram");
    unsigned int indices[8] = {
            0,1,
            1,2,
            2,3,
            3,0
    };

    void prepareFrame(GLenum target, GLuint id) {
        GLuint _frameBuffer;
        glGenFramebuffers(1, &_frameBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, _frameBuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target, id, 0);
    }

    float *prepareCoordinates(float *vertices) {
        return new float[] {
                //clock-wise , 0,0 is at center
                vertices[0],vertices[1],  //TopLeft
                vertices[2], vertices[1], //TopRight
                vertices[2], vertices[3], //BottomRight
                vertices[0], vertices[3]  //BottomLeft
        };
    }

public:

    BoundingBoxUtil() {
        _rectangleProgram.loadVsPs(vertexShaderSource,fragmentShaderSource);
    }

    ~BoundingBoxUtil() {
        _rectangleProgram.cleanup();
    }

    void drawBoundingBox(GLenum target, GLuint id, float *vertices) {
        // Set the viewport as it gets altered by inference engine.
        glViewport(0, 0, widthDim, heightDim);
        prepareFrame(target,id);
        float *coordinates = prepareCoordinates(vertices);
        unsigned int VBO,EBO;
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(coordinates[0]), coordinates, GL_STATIC_DRAW);

        glGenBuffers(1,&EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        _rectangleProgram.use();
        glLineWidth(3.0f);
        glDrawElements(GL_LINES, 8, GL_UNSIGNED_INT, 0);

        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        delete coordinates;
    }
};

#endif //SNNDEMO_BOUNDINGBOXUTIL_H
