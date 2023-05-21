package main

import "core:time"
import "core:image/png"
import "core:bytes"
import "core:mem"
import "core:slice"
import "core:fmt"
import glm "core:math/linalg/glsl"
import gl "vendor:OpenGL"
import sdl "vendor:sdl2"

WINDOW_WIDTH :: 1280
WINDOW_HEIGHT :: 720
MAT_DEFAULT :: glm.mat4{
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
}
DEFAULT_VERT_SRC :: `#version 330 core
layout(location=0) in vec3 a_position;
layout(location=1) in vec2 a_texcoord0;

uniform mat4 u_transform;
out vec2 v_texcoord0;

void main() {
    gl_Position = u_transform * vec4(a_position, 1.0);
    v_texcoord0 = a_texcoord0;
}
`

DEFAULT_FRAG_SRC :: `#version 330 core
in vec2 v_texcoord0;
out vec4 o_color;

uniform sampler2D tex;
uniform vec4 u_color;

void main() {
    vec4 tex_color = texture(tex, v_texcoord0);
	o_color = tex_color * u_color;
}
`

Mesh :: struct {
    vao, vbo, ebo: u32,
}

Transform :: struct {
    position: glm.vec3,
    rotation: glm.vec3,
    scale: glm.vec3,
}

Sprite :: struct {
    transform: Transform,
    mesh: Mesh,
    color: glm.vec4,
    tex: u32,
}

@(private)
create_tex :: proc(filepath: string) -> u32 {
    tex: u32 = 0
    img, err := png.load_from_file(filepath, {}, context.temp_allocator) // Allocate to ring buffer
    assert(err == nil, "Error loading PNG")

    gl.GenTextures(1, &tex)
    gl.BindTexture(gl.TEXTURE_2D, tex)
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT)
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT)
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR)
    gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
    gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, i32(img.width), i32(img.height), 0, gl.RGBA, gl.UNSIGNED_BYTE, &img.pixels.buf[img.pixels.off])
    gl.GenerateMipmap(gl.TEXTURE_2D)
    gl.BindTexture(gl.TEXTURE_2D, 0)
    return tex
}

@(private)
create_quad :: proc() -> Mesh {
    // These are constants
    QUAD_VERTS: []f32 : {
        // Vertex           UV            Color
        -0.5, -0.5, 0.0,    0.0, 0.0,     1.0, 0.0, 0.0, 1.0,
        0.5, -0.5, 0.0,     1.0, 0.0,     0.0, 1.0, 0.0, 1.0,
        -0.5, 0.5, 0.0,     0.0, 1.0,     0.0, 0.0, 1.0, 1.0,
        0.5, 0.5, 0.0,      1.0, 1.0,     1.0, 1.0, 0.0, 1.0,
    }
    QUAD_IDXS: []u32 : {
        0, 1, 2,
        1, 3, 2,
    }

    mesh := Mesh{}
    gl.GenVertexArrays(1, &mesh.vao)
    gl.GenBuffers(1, &mesh.vbo)
    gl.GenBuffers(1, &mesh.ebo)

    gl.BindVertexArray(mesh.vao)

    gl.BindBuffer(gl.ARRAY_BUFFER, mesh.vbo)
    gl.BufferData(gl.ARRAY_BUFFER, len(QUAD_VERTS) * size_of(f32), raw_data(QUAD_VERTS), gl.STATIC_DRAW)

    gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, mesh.ebo)
    gl.BufferData(gl.ELEMENT_ARRAY_BUFFER, len(QUAD_IDXS) * size_of(u32), raw_data(QUAD_IDXS), gl.STATIC_DRAW)

    gl.EnableVertexAttribArray(0)
    gl.EnableVertexAttribArray(1)
    gl.EnableVertexAttribArray(2)
    gl.VertexAttribPointer(0, 3, gl.FLOAT, false, size_of(f32) * 9, 0)
    gl.VertexAttribPointer(1, 2, gl.FLOAT, false, size_of(f32) * 9, size_of(f32) * 3)
    gl.VertexAttribPointer(2, 4, gl.FLOAT, false, size_of(f32) * 9, size_of(f32) * 5)

    gl.BindBuffer(gl.ARRAY_BUFFER, 0)
    gl.BindVertexArray(0)
    gl.BindBuffer(gl.ELEMENT_ARRAY_BUFFER, 0)

    return mesh
}

@(private)
sprite_create :: #force_inline proc(transform: Transform, color: glm.vec4, tex: u32) -> Sprite {
    return Sprite{
        transform = transform,
        mesh = create_quad(),
        color = color,
        tex = tex,
    }
}

@(private)
sprite_destroy :: #force_inline proc(sprite: ^Sprite) {
    gl.DeleteBuffers(1, &sprite.mesh.vbo)
    gl.DeleteBuffers(1, &sprite.mesh.ebo)
    gl.DeleteVertexArrays(1, &sprite.mesh.vao)
}

@(private)
sprite_render :: proc(matProj, matView, matModel, matTransform: ^glm.mat4,
                      uniforms: ^gl.Uniforms,
                      sprite: ^Sprite) {
    matModel^ = MAT_DEFAULT *
               glm.mat4Translate(sprite.transform.position) *
               glm.mat4Rotate(glm.vec3{1.0, 0.0, 0.0}, glm.radians(sprite.transform.rotation.x)) *
               glm.mat4Rotate(glm.vec3{0.0, 1.0, 0.0}, glm.radians(sprite.transform.rotation.y)) *
               glm.mat4Rotate(glm.vec3{0.0, 0.0, 1.0}, glm.radians(sprite.transform.rotation.z)) *
               glm.mat4Scale(sprite.transform.scale)

    // Do this on the CPU for per object, on the GPU this would happen per vertex
    matTransform^ = matProj^ * matView^ * matModel^
    gl.UniformMatrix4fv(uniforms["u_transform"].location, 1, false, &(matTransform^)[0, 0])
    gl.Uniform4fv(uniforms["u_color"].location, 1, &sprite.color[0])

    // As we've bound the vert/idx buffers to a vertex array object, we can simply bind it and call DrawElements (as long as we know the index count)
    gl.BindVertexArray(sprite.mesh.vao)
    gl.DrawElements(gl.TRIANGLES, 6, gl.UNSIGNED_INT, nil)
}

main :: proc() {
    when ODIN_DEBUG {
        track: mem.Tracking_Allocator
        mem.tracking_allocator_init(&track, context.allocator)
        context.allocator = mem.tracking_allocator(&track)
        defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not freed: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			if len(track.bad_free_array) > 0 {
				fmt.eprintf("=== %v incorrect frees: ===\n", len(track.bad_free_array))
				for entry in track.bad_free_array {
					fmt.eprintf("- %p @ %v\n", entry.memory, entry.location)
				}
			}
		}
    }

    assert(sdl.Init(sdl.INIT_VIDEO) == 0, "Failed to initialize SDL")
    defer sdl.Quit()

    window := sdl.CreateWindow("Render Test",
                               sdl.WINDOWPOS_CENTERED,
                               sdl.WINDOWPOS_CENTERED,
                               WINDOW_WIDTH, WINDOW_HEIGHT,
                               { sdl.WindowFlag.OPENGL })
    assert(window != nil, sdl.GetErrorString())
    defer sdl.DestroyWindow(window)

    glCtx := sdl.GL_CreateContext(window)
    defer sdl.GL_DeleteContext(glCtx)
    gl.load_up_to(3, 3, sdl.gl_set_proc_address)

    defaultShader, ok := gl.load_shaders_source(DEFAULT_VERT_SRC, DEFAULT_FRAG_SRC)
    assert(ok, "Error loading default shader")
    uniforms := gl.get_uniforms_from_program(defaultShader)
    defer gl.destroy_uniforms(uniforms)

    // Load a texture
    texUfo: u32 = create_tex("../assets/ufo.png")
    defer gl.DeleteTextures(1, &texUfo)

    // Create a bunch of sprites (with unique meshes because I'm lazy)
    sprites: []Sprite = {
        sprite_create(
            Transform{{0.0, -0.5, -0.1}, {0.0, 0.0, 0.0}, {2.0, 1.0, 1.0}},
            {0.5, 0.7, 1.0, 1.0},
            texUfo,
        ),
        sprite_create(
            Transform{{-1.0, 0.5, 0.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}},
            {1.0, 1.0, 1.0, 1.0},
            texUfo,
        ),
        sprite_create(
            Transform{{1.0, 0.5, 0.0}, {0.0, 0.0, 0.0}, {0.5, 3.0, 0.5}},
            {1.0, 0.5, 0.5, 1.0},
            texUfo,
        ),
    }
    defer {
        for _,idx in sprites {
            sprite_destroy(&sprites[idx])
        }
    }

    // Define some vars for simulation
    tick         := time.tick_now()

    camTransform := Transform{{0.0, 0.0, -5.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}
    camDir       := f32(1.0)
    fov          := f32(70.0)
    aspectRatio  := f32(WINDOW_WIDTH) / f32(WINDOW_HEIGHT)

    matModel     := MAT_DEFAULT
    matView      := MAT_DEFAULT
    matProj      := glm.mat4Perspective(fov, aspectRatio, 0.1, 100.0)
    matTransform := MAT_DEFAULT

    // NOTE: As our sprites are all transparent, we need to sort them based on distance from 'viewport'
    // Normally done per frame based on the distance from camera, but here we can cheap out and just calculate once using position Z difference
    slice.sort_by_cmp(sprites, proc(a, b: Sprite) -> slice.Ordering {
        return a.transform.position.z > b.transform.position.z ? .Less :
               a.transform.position.z < b.transform.position.z ? .Greater :
               .Equal
    })

    loop: for {

        /**************************************
        *   INPUT
        **************************************/

        event: sdl.Event
        for sdl.PollEvent(&event) {
            #partial switch event.type {
                case .QUIT: break loop
            }
        }

        dt := f32(f64(time.tick_since(tick)) / f64(time.Second))
        tick = time.tick_now()

        /**************************************
        *   UPDATE
        **************************************/

        sprites[0].transform.rotation.y += 50.0 * dt
        sprites[1].transform.rotation.y -= 50.0 * dt
        if camTransform.position.z <= -5.0 { camDir = 1.0 }
        if camTransform.position.z >= -2.0 { camDir = -1.0 }
        camTransform.position.z += 2.0 * dt * camDir

        /**************************************
        *   RENDER
        **************************************/

        // This is done once per frame
        matView = MAT_DEFAULT *
                  glm.mat4Translate(camTransform.position) *
                  glm.mat4Rotate(glm.vec3{1.0, 0.0, 0.0}, glm.radians(camTransform.rotation.x)) *
                  glm.mat4Rotate(glm.vec3{0.0, 1.0, 0.0}, glm.radians(camTransform.rotation.y)) *
                  glm.mat4Rotate(glm.vec3{0.0, 0.0, 1.0}, glm.radians(camTransform.rotation.z))

        gl.Viewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        gl.ClearColor(0.0, 0.0, 0.0, 1.0)
        gl.Clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT)

        // Shaders are often batched for multiple objects
        gl.UseProgram(defaultShader)
        gl.Enable(gl.BLEND)
        gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

        // This is done for every object per frame
        for _,idx in sprites {
            sprite := sprites[idx]
            gl.BindTexture(gl.TEXTURE_2D, sprite.tex)
            sprite_render(&matProj, &matView, &matModel, &matTransform,
                          &uniforms,
                          &sprite)
        }

        /**************************************
        *   FLUSH
        **************************************/

        sdl.GL_SwapWindow(window)
    }
}
