use cgmath::{Deg, Rad, Matrix4, Point3, Vector3, Vector4};
use closure::closure;
use erupt::{
    cstr,
    utils::{self, surface},
    vk, DeviceLoader, EntryLoader, InstanceLoader,
    vk::{Device, MemoryMapFlags},
};
use winit::{
    dpi::PhysicalSize,
    event::{
        Event, KeyboardInput, WindowEvent,
        ElementState, StartCause, VirtualKeyCode,
        DeviceEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    window::Window
};
use nalgebra_glm as glm;
use std::{
    ffi::{c_void, CStr, CString},
    fs,
    fs::{write, OpenOptions},
    io::prelude::*,
    mem::*,
    // sync::mpsc::channel,
    os::raw::c_char,
    ptr,
    result::Result,
    result::Result::*,
    string::String,
    sync::{Arc, Mutex, MutexGuard, mpsc, mpsc::{channel}},
    thread,
    time,
};
use structopt::StructOpt;

unsafe extern "system" fn debug_callback(
    _message_severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let str_99 = String::from(CStr::from_ptr((*p_callback_data).p_message).to_string_lossy());
    eprintln!(
        "{}",
        CStr::from_ptr((*p_callback_data).p_message).to_string_lossy()
    );
    vk::FALSE
}

const TITLE: &str = "vulkan-routine";
const FRAMES_IN_FLIGHT: usize = 3;
const LAYER_KHRONOS_VALIDATION: *const c_char = cstr!("VK_LAYER_KHRONOS_validation");

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long)]
    validation_layers: bool,
}

pub unsafe fn routine
()
{
    println!("Routine.");

    let opt = Opt { validation_layers: false };
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(TITLE)
        .with_resizable(true)
        .with_maximized(true)
        .build(&event_loop)
        .unwrap();

        let entry = EntryLoader::new().unwrap();
        let application_name = CString::new("Erupt Routine 100").unwrap();
        let engine_name = CString::new("Peregrine").unwrap();
        let app_info = vk::ApplicationInfoBuilder::new()
            .application_name(&application_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));


    let mut instance_extensions = surface::enumerate_required_extensions(&window).unwrap();
    if opt.validation_layers {
        instance_extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    let mut instance_layers = Vec::new();
    if opt.validation_layers {
        instance_layers.push(LAYER_KHRONOS_VALIDATION);
    }
    let device_extensions = vec![
        vk::KHR_SWAPCHAIN_EXTENSION_NAME,
        // vk::KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        vk::KHR_RAY_QUERY_EXTENSION_NAME,
        vk::KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        vk::KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        vk::KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME,
        vk::KHR_SPIRV_1_4_EXTENSION_NAME,
        vk::KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        vk::EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
    ];
    let mut device_layers = Vec::new();
    if opt.validation_layers {
        device_layers.push(LAYER_KHRONOS_VALIDATION);
    }
    let instance_info = vk::InstanceCreateInfoBuilder::new()
        .application_info(&app_info)
        .enabled_extension_names(&instance_extensions)
        .enabled_layer_names(&instance_layers);
    let instance = Arc::new(InstanceLoader::new(&entry, &instance_info).unwrap());
    let messenger = if opt.validation_layers {
        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXTBuilder::new()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING_EXT
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR_EXT,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL_EXT
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION_EXT
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE_EXT,
            )
            .pfn_user_callback(Some(debug_callback));

        instance.create_debug_utils_messenger_ext(&messenger_info, None).expect("problem creating debug util.");
    } else {
        Default::default()
    };
    let surface = surface::create_surface(&instance, &window, None).unwrap();

    let (physical_device, queue_family, format, present_mode, device_properties) = instance.enumerate_physical_devices(None)
    .unwrap()
    .into_iter()
    .filter_map(|physical_device| {
        let queue_family = match instance
            .get_physical_device_queue_family_properties(physical_device, None)
            .into_iter()
            .enumerate()
            .position(|(i, queue_family_properties)| {
                queue_family_properties
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                    && instance
                        .get_physical_device_surface_support_khr(
                            physical_device,
                            i as u32,
                            surface,
                        )
                        .unwrap()
            }) {
            Some(queue_family) => queue_family as u32,
            None => return None,
        };
        let formats = instance
            .get_physical_device_surface_formats_khr(physical_device, surface, None)
            .unwrap();
        let format = match formats
            .iter()
            .find(|surface_format| {
                (surface_format.format == vk::Format::B8G8R8A8_SRGB
                    || surface_format.format == vk::Format::R8G8B8A8_SRGB)
                    && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR_KHR
            })
            .or_else(|| formats.get(0))
        {
            Some(surface_format) => *surface_format,
            None => return None,
        };
        let present_mode = instance
            .get_physical_device_surface_present_modes_khr(physical_device, surface, None)
            .unwrap()
            .into_iter()
            .find(|present_mode| present_mode == &vk::PresentModeKHR::MAILBOX_KHR)
            .unwrap_or(vk::PresentModeKHR::FIFO_KHR);
        let supported_device_extensions = instance
            .enumerate_device_extension_properties(physical_device, None, None)
            .unwrap();
        let device_extensions_supported =
            device_extensions.iter().all(|device_extension| {
                let device_extension = CStr::from_ptr(*device_extension);

                supported_device_extensions.iter().any(|properties| {
                    CStr::from_ptr(properties.extension_name.as_ptr()) == device_extension
                })
            });
        if !device_extensions_supported {
            return None;
        }
        let device_properties = instance.get_physical_device_properties(physical_device);
        Some((
            physical_device,
            queue_family,
            format,
            present_mode,
            device_properties,
        ))
    })
    .max_by_key(|(_, _, _, _, properties)| match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 2,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => 0,
    })
    .expect("No suitable physical device found");


    println!("\n\n\nUsing physical device: {:?}\n\n\n", CStr::from_ptr(device_properties.device_name.as_ptr()));


    let queue_info = vec![vk::DeviceQueueCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let features = vk::PhysicalDeviceFeaturesBuilder::new();
    let device_info = vk::DeviceCreateInfoBuilder::new()
        .queue_create_infos(&queue_info)
        .enabled_features(&features)
        .enabled_extension_names(&device_extensions)
        .enabled_layer_names(&device_layers);
    let device  = Arc::new(Mutex::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap()));

    let mut device_locked = device.lock().unwrap();
    let queue = device_locked.get_device_queue(queue_family, 0);

    let surface_caps = instance.get_physical_device_surface_capabilities_khr(physical_device, surface).unwrap();
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    let swapchain_image_extent = match surface_caps.current_extent {
        vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        } => {
            let PhysicalSize { width, height } = window.inner_size();
            // vk::Extent2D { width: 4000, height: 4000 }
            vk::Extent2D { width, height }
        }
        normal => normal,
    };

    let swapchain_info = vk::SwapchainCreateInfoKHRBuilder::new()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(swapchain_image_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::default());

        let swapchain = device_locked.create_swapchain_khr(&swapchain_info, None).unwrap();
        let swapchain_images = device_locked.get_swapchain_images_khr(swapchain, None).unwrap();
        let swapchain_image_views: Vec<_> = swapchain_images
            .iter()
            .map(|swapchain_image| {
                let image_view_info = vk::ImageViewCreateInfoBuilder::new()
                    .image(*swapchain_image)
                    .view_type(vk::ImageViewType::_2D)
                    .format(format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(
                        vk::ImageSubresourceRangeBuilder::new()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    );
                device_locked.create_image_view(&image_view_info, None).unwrap()
            })
            .collect();
    
        let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
                .queue_family_index(queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = device_locked.create_command_pool(&command_pool_info, None).unwrap();

}

