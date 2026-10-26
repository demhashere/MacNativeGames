/* The game window on Linux (linux/frontend_sdl.c): SDL3, a Vulkan surface for the NVN backend. */
#pragma once
#include <stdint.h>
#include <SDL3/SDL.h>
#include <vulkan/vulkan.h>

void front_sdl_post(void (*fn)(void));                 /* run fn on the main (event) thread */
SDL_Window *front_sdl_window(void);
const char *const *front_vk_instance_extensions(uint32_t *count);
int front_vk_create_surface(VkInstance instance, VkSurfaceKHR *surface);   /* on the main thread; 1 = made */
void front_drawable_size(uint32_t *w, uint32_t *h);   /* the window's size in pixels */
int front_display_hz(void);                           /* refresh rate of the window's display (60 if unknown) */
void front_frame_presented(void);
const char *front_boot_line(char *out, size_t n);     /* boot status text while the game has shown no frame, else NULL */
void gamepad_sdl_event(const SDL_Event *e);           /* linux/gamepad_sdl.c */
