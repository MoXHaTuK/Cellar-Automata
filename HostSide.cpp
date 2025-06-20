#include "HostSide.hpp"
#include <tiffio.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

extern void launchPhasePut(uint8_t depth, void*, void*, int, int);
extern void launchPhaseSetLen(uint8_t depth, void*, void*, int, int);
extern void launchPhaseSetSpd(uint8_t depth, void*, void*, int, int);
extern void launchPhaseLights(uint8_t depth, void*, void*, int, int);
extern void launchPhaseTurn(uint8_t depth, void*, void*, int, int);
extern void launchPhaseMove(uint8_t depth, void*, void*, int, int);

bool saveTiff(const std::string& file, const Frame& f)
{
    if (!(f.bitDepth == 8 || f.bitDepth == 16)) {
        std::cerr << "saveTiff: bitDepth must be 8 or 16\n";
        return false;
    }
    if (!(f.channels == 1 || f.channels == 3)) {
        std::cerr << "saveTiff: only 1-channel (gray) or 3-channel (RGB) supported\n";
        return false;
    }
    if (f.data() == nullptr) {
        std::cerr << "saveTiff: empty data buffer\n";
        return false;
    }

    TIFF* t = TIFFOpen(file.c_str(), "w");
    if (!t) {
        std::cerr << "saveTiff: TIFFOpen failed for '" << file << "'\n";
        return false;
    }

    TIFFSetField(t, TIFFTAG_IMAGEWIDTH, f.w);
    TIFFSetField(t, TIFFTAG_IMAGELENGTH, f.h);
    TIFFSetField(t, TIFFTAG_SAMPLESPERPIXEL, f.channels);
    TIFFSetField(t, TIFFTAG_BITSPERSAMPLE, f.bitDepth);
    TIFFSetField(t, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(t, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

    uint16_t photometric = (f.channels == 1)
        ? PHOTOMETRIC_MINISBLACK
        : PHOTOMETRIC_RGB;
    TIFFSetField(t, TIFFTAG_PHOTOMETRIC, photometric);

    const size_t bytesPerLine = size_t(f.w) * f.channels * (f.bitDepth / 8);
    for (int y = 0; y < f.h; ++y)
    {
        const void* srcLine =
            (f.bitDepth == 8)
            ? static_cast<const void*>(&f.data8[y * f.w * f.channels])
            : static_cast<const void*>(&f.data16[y * f.w * f.channels]);

        if (TIFFWriteScanline(t, const_cast<void*>(srcLine), y, 0) < 0) {
            std::cerr << "saveTiff: TIFFWriteScanline failed (row " << y << ")\n";
            TIFFClose(t);
            return false;
        }
    }

    TIFFClose(t);
    return true;
}

bool loadTiff(const std::string& file, Frame& f)
{
    TIFF* t = TIFFOpen(file.c_str(), "r");
    if (!t) {
        std::cerr << "TIFFOpen: cannot open '" << file << "'\n";
        return false;
    }

    uint32_t w = 0, h = 0;
    uint16_t spp = 1, bps = 8, planar = PLANARCONFIG_CONTIG;
    uint16_t photo = 0;

    TIFFGetField(t, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(t, TIFFTAG_IMAGELENGTH, &h);
    TIFFGetFieldDefaulted(t, TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(t, TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(t, TIFFTAG_PLANARCONFIG, &planar);
    TIFFGetFieldDefaulted(t, TIFFTAG_PHOTOMETRIC, &photo);

    if (planar != PLANARCONFIG_CONTIG) {
        std::cerr << "Unsupported planar config\n";
        TIFFClose(t); return false;
    }
    if (bps != 8 && bps != 16) {
        std::cerr << "Unsupported bit depth: " << bps << "\n";
        TIFFClose(t); return false;
    }
    if (!(spp == 1 || spp == 3)) {
        std::cerr << "Unsupported channels: " << spp << "\n";
        TIFFClose(t); return false;
    }

    f.w = (int)w;
    f.h = (int)h;
    f.channels = spp;
    f.bitDepth = bps;

    size_t count = size_t(w) * h * spp;
    if (bps == 8)  f.data8.resize(count);
    else           f.data16.resize(count);

    tsize_t scanlineSize = TIFFScanlineSize(t);
    std::vector<uint8_t>  buf8;
    std::vector<uint16_t> buf16;

    if (bps == 8)  buf8.resize(scanlineSize);
    else           buf16.resize(scanlineSize / 2);  // 2 байта на сэмпл

    for (uint32_t y = 0; y < h; ++y)
    {
        void* dst = (bps == 8) ? (void*)buf8.data() : (void*)buf16.data();
        if (TIFFReadScanline(t, dst, y, 0) < 0) {
            std::cerr << "TIFFReadScanline failed\n";
            TIFFClose(t); return false;
        }

        if (bps == 8) {
            std::memcpy(&f.data8[y * w * spp], buf8.data(), w * spp);
        }
        else {
            std::memcpy(&f.data16[y * w * spp], buf16.data(), w * spp * sizeof(uint16_t));
        }
    }

    TIFFClose(t);
    return true;
}

static GLFWwindow* win = nullptr;
static GLuint      tex = 0;
static GLuint      pbo = 0;
static GLuint      vao = 0;
static GLuint      vbo = 0;
static GLuint      prog = 0;

static GLuint compile(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << '\n';
        std::exit(EXIT_FAILURE);
    }
    return s;
}

static GLuint buildProgram(const char* vsSrc, const char* fsSrc)
{
    GLuint vs = compile(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fsSrc);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetProgramInfoLog(p, 512, nullptr, log);
        std::cerr << "Program link error:\n" << log << '\n';
        std::exit(EXIT_FAILURE);
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

void initGL(int w, int h)
{
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (!glfwInit()) { std::cerr << "GLFW init failed\n"; std::exit(EXIT_FAILURE); }

    win = glfwCreateWindow(w, h, "Traffic CA", nullptr, nullptr);
    if (!win) { std::cerr << "Window creation failed\n"; std::exit(EXIT_FAILURE); }
    glfwMakeContextCurrent(win);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "GLEW init failed\n"; std::exit(EXIT_FAILURE); }
    glGetError();

    std::cout << "OpenGL: " << glGetString(GL_VERSION) << "\n"
        << "GPU    : " << glGetString(GL_RENDERER) << "\n";

    glViewport(0, 0, w, h);
    glClearColor(0, 0, 0, 1);

    glCreateTextures(GL_TEXTURE_2D, 1, &tex);
    glTextureStorage2D(tex, 1, GL_R8, w, h); 
    glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glCreateBuffers(1, &pbo);
    glNamedBufferData(pbo, w * h, nullptr, GL_STREAM_DRAW);

    static const char* vsSrc = R"(#version 450 core
        const vec2 verts[6] = vec2[6](
             vec2(-1,-1), vec2( 1,-1), vec2( 1, 1),
             vec2(-1,-1), vec2( 1, 1), vec2(-1, 1));
        const vec2 uvs[6] = vec2[6](
             vec2(0,0), vec2(1,0), vec2(1,1),
             vec2(0,0), vec2(1,1), vec2(0,1));
        out vec2 vUV;
        void main(){
            gl_Position = vec4(verts[gl_VertexID],0,1);
            vUV = uvs[gl_VertexID];
        })";

    static const char* fsSrc = R"(#version 450 core
        in  vec2 vUV;
        uniform sampler2D uTex;
        out vec4 FragColor;
        void main() {
            float r = texture(uTex, vUV).r;
            FragColor = vec4(r,r,r,1);
        })";

    prog = buildProgram(vsSrc, fsSrc);

    glCreateVertexArrays(1, &vao);
}

void updateFrame(const Frame& f)
{
    if (!win) return;

    const size_t bytes = f.byteSize();
    glNamedBufferSubData(pbo, 0, bytes, f.data());

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    GLenum type = (f.bitDepth == 8) ? GL_UNSIGNED_BYTE : GL_UNSIGNED_SHORT;
    GLenum format = GL_RED;

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glTextureSubImage2D(tex, 0, 0, 0, f.w, f.h, format, type, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

bool renderFrame()
{
    if (!win) return false;
    if (glfwWindowShouldClose(win)) return false;

    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    glBindTextureUnit(0, tex);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glfwSwapBuffers(win);
    glfwPollEvents();
    return true;
}