use cgmath::{Deg, Rad, Matrix4, Point3, Vector3, Vector4};
use closure::closure;
use erupt::{
    cstr,
    extensions::khr_surface::{SurfaceKHR, SurfaceFormatKHR, PresentModeKHR},
    extensions::khr_swapchain::SwapchainKHR,
    SmallVec,
    utils::{self, surface},
    vk, DeviceLoader, EntryLoader, InstanceLoader,
    vk::{Device, MemoryMapFlags, Image},
};
use memoffset::offset_of;
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

const SHADER_VERT: &[u8] = include_bytes!("../spv/terrain_100.vert.spv");
const SHADER_FRAG: &[u8] = include_bytes!("../spv/terrain_100.frag.spv");

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

    // let window = WindowBuilder::new()
    //     .with_title(TITLE)
    //     .with_resizable(true)
    //     // .with_maximized(true)
    //     .with_inner_size(winit::dpi::LogicalSize::new(2000, 2000))
    //     .build(&event_loop)
    //     .unwrap();

    let window = Arc::new(
        WindowBuilder::new()
            .with_title(TITLE)
            .with_resizable(true)
            // .with_maximized(true)
            .with_inner_size(winit::dpi::LogicalSize::new(2000, 2000))
            .build(&event_loop)
            .unwrap()
    );


    let entry = EntryLoader::new().unwrap();
    let application_name = CString::new("Erupt Routine 100").unwrap();
    let engine_name = CString::new("Peregrine").unwrap();
    let app_info = vk::ApplicationInfoBuilder::new()
        .application_name(&application_name)
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 1, 0, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));


    let mut instance_extensions = surface::enumerate_required_extensions(&*window).unwrap();
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
    let surface = surface::create_surface(&instance, &*window, None).unwrap();

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
    // let device  = Arc::new(Mutex::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap()));
    let device  = Arc::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap());

    // let mut dee = device.lock().unwrap();
    let queue = device.get_device_queue(queue_family, 0);

    let surface_caps = instance.get_physical_device_surface_capabilities_khr(physical_device, surface).unwrap();
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }
    

    let present_mode = Arc::new(present_mode);
    let format = Arc::new(format);
    let surface = Arc::new(surface);
    let physical_device = Arc::new(physical_device);
    // let window2 = Arc::new(window);

    let (
        swapchain, 
        swapchain_images, 
        swapchain_image_views,
        swapchain_image_extent,
    ) = create_swapchain(
        device.clone(),
        instance.clone(),
        present_mode.clone(),
        format.clone(),
        surface.clone(),
        physical_device.clone(),
        window.clone(),
    ).unwrap();

    // At this point we have swapchain.

    //https://www.intel.com/content/www/us/en/developer/articles/training/practical-approach-to-vulkan-part-1.html
    // "": Vulkan applications:
    // Acquire a swapchain image, render into the acquired image, present the image to the screen, repeat.
    // To complete any task we need a command buffer, these are grouped into pools.
    // Acquisition of an image to render into involves a semaphore or a fence.


    // ""
    // At least one command buffer. This buffer is required to record and submit rendering commands to the graphics hardware.
    // Image available (image acquired) semaphore. This semaphore gets signaled by a presentation engine and is used to synchronize rendering with swap chain image acquisition.
    // Rendering finished (ready to present) semaphore. This semaphore gets signaled when processing of the command buffer is finished and is used to synchronize presentation with the rendering process.
    // Fence. A fence indicates when the rendering of a whole frame is finished and informs our application when it can reuse resources from a given frame.

    let num_threads = 3; // for now.

    for i_t in 0..(num_threads - 1) {
        // 
    }
    // each frame in each thread gets its 
    

    // The Intel article recommends three sets of frame resources.
    // I guess these are one-to-one with the threads times frames.
    // I guess we use the 3 instead of the frames number, they are somewhat though not completely related.

    // a command pool then would be part of frame resources.

    // 3 sets of frame resources per thread.

    // let frame_resources = 

    //https://stackoverflow.com/questions/53438692/creating-multiple-command-pools-per-thread-in-vulkan
    // One command pool per frame per thread.
    // So multiple threads just for rendering.  Plus the state management thread(s).

    //https://stackoverflow.com/questions/73502161/vulkan-why-have-multiple-command-buffers-per-pool
    // Multiple command buffers per pool.


    // Question: Do semaphores and fences need to be shared across threads? Or can threads maintain independent sets?

    // After we do the structure foundation as above, we can focus on the render loop and render structure.
    // Attachments, pipeline, depth, shader config, vertex buffers, index buffers
    // swapchain framebuffers 
    // pipeline, swapchin framebuffer, shader objects,
    // then we operate on command buffers in multiple threads, setup window event loop, draws and state updates.



    let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family);

    // let frames_resources: Vec<FrameResources> = (0..3)
    // .map(|_| {
    //     FrameResources {
    //         name: vec![0, 3, 7],
    //         ias: [ // image available semaphores
    //             device.create_semaphore(&semaphore_info, None).unwrap(),
    //             device.create_semaphore(&semaphore_info, None).unwrap(),
    //             device.create_semaphore(&semaphore_info, None).unwrap(),
    //         ],
    //         rfs: [ // render finished semaphores
    //             device.create_semaphore(&semaphore_info, None).unwrap(),
    //             device.create_semaphore(&semaphore_info, None).unwrap(),
    //             device.create_semaphore(&semaphore_info, None).unwrap(),
    //         ],
    //         iff: [ // in flight fences
    //             device.create_fence(&fence_info, None).unwrap(),
    //             device.create_fence(&fence_info, None).unwrap(),
    //             device.create_fence(&fence_info, None).unwrap(),
    //         ],
    //         command_pools: [
    //             device.create_command_pool(&command_pool_info, None).unwrap(),
    //             device.create_command_pool(&command_pool_info, None).unwrap(),
    //             device.create_command_pool(&command_pool_info, None).unwrap(),
    //         ]
    //     }
    // })
    // .collect();

    // println!("frames_resources.len() {}", frames_resources.len());

    // There are shared resources between threads. Game state.  Buffer references.
    // Resources above are independent per thread.

    // So now we need something to render to an image.  Vertex buffers, index buffers for a mesh
    // materials, textures, transforms on objects in R^3.  Uniform buffers.

    // What to render?  Akshually maybe the terrain model again, testing if we can optimize beyond
    // the the disappointing frame rate in vulkan_8700, which is multi-threaded but likely contains 
    // Vulkan anti-patterns.

    // We can improve the matrix transforms and perspective also.  Space transforms.

    // render-pass, pipeline, descriptor pools, set layouts, uniform buffer object
    // cameras, vertex objects
    // attachments.
    // dependencies, 

    // swapchain_framebuffers // these are re-used or no?
    // command buffers this is ad-hoc as command pools are reset

    let (mut vertices_terr, mut indices_terr) = load_model().unwrap();
    let indices_length = Arc::new(indices_terr.len());
    let physical_device_memory_properties = instance.get_physical_device_memory_properties(*physical_device);

    let info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let tcp = device.create_command_pool(&command_pool_info, None).unwrap();


    let vertex_buffer = Arc::new(
        buffer_vertices(
            device.clone(),
            queue,
            tcp,
            &mut vertices_terr,
        ).unwrap()
    );

    let index_buffer = Arc::new(
        buffer_indices(
            device.clone(),
            queue,
            tcp,
            &mut indices_terr,
        ).unwrap()
    );
    
    // Uniform buffer 
    let info = vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new()
        .binding_flags(&[vk::DescriptorBindingFlags::empty()]);
    let samplers = [vk::Sampler::default()];
    let binding = vk::DescriptorSetLayoutBindingBuilder::new()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .immutable_samplers(&samplers);
    let bindings = &[binding];
    let info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty()) 
        .bindings(bindings);
    let descriptor_set_layout = device.create_descriptor_set_layout(&info, None).unwrap();
    let ubo_size = ::std::mem::size_of::<UniformBufferObject>();
    let mut uniform_buffers: Vec<vk::Buffer> = vec![];
    let mut uniform_buffers_memories: Vec<vk::DeviceMemory> = vec![];
    let swapchain_image_count = swapchain_images.len();
    let (uniform_buffer, uniform_buffer_memory) = create_buffer(
        device.clone(),
        ubo_size as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        2,
    );
    // These will be multiplied by 3 per thread with FrameResources
    let uniform_buffer = Arc::new(Mutex::new(
        uniform_buffer,
    ));
    let uniform_buffer_memory = Arc::new(Mutex::new(
        uniform_buffer_memory,
    ));
    

    
// https://www.intel.com/content/www/us/en/developer/articles/training/api-without-secrets-introduction-to-vulkan-part-4.html

    // Have vertex and index buffers, uniforms partly setup
    // Vaguely recall need to set up render pass,
    // pipeline, attachments, descriptor sets
    // starting from the descriptor pool building line
    // 506 in 8700.
    let pool_size = vk::DescriptorPoolSizeBuilder::new()
        ._type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(3 as u32);
    let pool_sizes = &[pool_size];
    let set_layouts = &[descriptor_set_layout];
    let pool_info = vk::DescriptorPoolCreateInfoBuilder::new()
        .pool_sizes(pool_sizes)
        .max_sets(3 as u32);

    let desc_pool = device.create_descriptor_pool(&pool_info, None).unwrap();
    let d_set_alloc_info = vk::DescriptorSetAllocateInfoBuilder::new()
        .descriptor_pool(desc_pool)
        .set_layouts(set_layouts);
    let descriptor_sets = device.allocate_descriptor_sets(&d_set_alloc_info).expect("failed in alloc DescriptorSet");
    let ubo_size = ::std::mem::size_of::<UniformBufferObject>() as u64;
    // We skip the for loop for now that updates the 
    // uniform buffers.
    let attachments = vec![
        vk::AttachmentDescriptionBuilder::new()
            .format(format.format)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        vk::AttachmentDescriptionBuilder::new()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlagBits::_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    ];
    let depth_attach_ref = vk::AttachmentReferenceBuilder::new()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let color_attachment_refs = vec![vk::AttachmentReferenceBuilder::new()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
    let subpasses = vec![vk::SubpassDescriptionBuilder::new()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_refs)
        .depth_stencil_attachment(&depth_attach_ref)];
    let dependencies = vec![vk::SubpassDependencyBuilder::new()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
    let render_pass_info = vk::RenderPassCreateInfoBuilder::new()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);  
    let render_pass = Arc::new(
        device.create_render_pass(&render_pass_info, None).unwrap()
    );
    // Now can create the pipeline:
    // pipeline_101
    let info = vk::DescriptorSetLayoutBindingFlagsCreateInfoBuilder::new()
        .binding_flags(&[vk::DescriptorBindingFlags::empty()]);
    let samplers = [vk::Sampler::default()];
    let binding = vk::DescriptorSetLayoutBindingBuilder::new()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .immutable_samplers(&samplers);
    let slice = &[binding];
    let info = vk::DescriptorSetLayoutCreateInfoBuilder::new()
        .flags(vk::DescriptorSetLayoutCreateFlags::empty()) 
        .bindings(slice);
    let descriptor_set_layout = device.create_descriptor_set_layout(&info, None).unwrap();
    let depth_image_info = vk::ImageCreateInfoBuilder::new()
        .flags(vk::ImageCreateFlags::empty())
        .image_type(vk::ImageType::_2D)
        .format(vk::Format::D32_SFLOAT)
        .extent(vk::Extent3D {
            width: swapchain_image_extent.width,
            height: swapchain_image_extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlagBits::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&[0])
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let depth_image = device.create_image(&depth_image_info, None)
        .expect("Failed to create depth (texture) Image.");   
    let dpth_img_mem_reqs = device.get_image_memory_requirements(depth_image);
    let dpth_img_mem_info = vk::MemoryAllocateInfoBuilder::new()
        .memory_type_index(1)
        .allocation_size(dpth_img_mem_reqs.size);
    let depth_image_memory = device.allocate_memory(&dpth_img_mem_info, None)
        .expect("Failed to alloc mem for depth image.");
    device.bind_image_memory(depth_image, depth_image_memory, 0)
        .expect("Failed to bind depth image memory.");
    let depth_image_view_info = vk::ImageViewCreateInfoBuilder::new()
        .flags(vk::ImageViewCreateFlags::empty())
        .image(depth_image)
        .view_type(vk::ImageViewType::_2D)
        .format(vk::Format::D32_SFLOAT)
        .components(vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        })
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        });
    let depth_image_view = Arc::new(
        device.create_image_view(&depth_image_view_info, None)
        .expect("Failed to create image view.")
    );
    let entry_point = CString::new("main").unwrap();
    let vert_decoded = utils::decode_spv(SHADER_VERT).unwrap();
    let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&vert_decoded);
    let shader_vert = device.create_shader_module(&module_info, None).unwrap();
    let frag_decoded = utils::decode_spv(SHADER_FRAG).unwrap();
    let module_info = vk::ShaderModuleCreateInfoBuilder::new().code(&frag_decoded);
    let shader_frag = device.create_shader_module(&module_info, None).unwrap();
    let shader_stages = vec![
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::VERTEX)
            .module(shader_vert)
            .name(&entry_point),
        vk::PipelineShaderStageCreateInfoBuilder::new()
            .stage(vk::ShaderStageFlagBits::FRAGMENT)
            .module(shader_frag)
            .name(&entry_point),
    ];
    let vertex_buffer_bindings_desc_info = vk::VertexInputBindingDescriptionBuilder::new()
        .binding(0)
        .stride(std::mem::size_of::<VertexV3>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vert_buff_att_desc_info_pos = vk::VertexInputAttributeDescriptionBuilder::new()
        .location(0)
        .binding(0)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .offset(offset_of!(VertexV3, pos) as u32,);
    let vert_buff_att_desc_info_color = vk::VertexInputAttributeDescriptionBuilder::new()
        .location(1)
        .binding(0)
        .format(vk::Format::R32G32B32A32_SFLOAT)
        .offset(offset_of!(VertexV3, color) as u32,);
    let vertex_input = vk::PipelineVertexInputStateCreateInfoBuilder::new()
        .flags(vk::PipelineVertexInputStateCreateFlags::empty())
        .vertex_binding_descriptions(&[vertex_buffer_bindings_desc_info])
        .vertex_attribute_descriptions(&[vert_buff_att_desc_info_pos, vert_buff_att_desc_info_color])
        .build_dangling();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfoBuilder::new()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewports = vec![vk::ViewportBuilder::new()
        .x(0.0)
        .y(0.0)
        .width(swapchain_image_extent.width as f32)
        .height(swapchain_image_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)];
    let scissors = vec![vk::Rect2DBuilder::new()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(*swapchain_image_extent)];
    let viewport_state = vk::PipelineViewportStateCreateInfoBuilder::new()
        .viewports(&viewports)
        .scissors(&scissors);
    let rasterizer = vk::PipelineRasterizationStateCreateInfoBuilder::new()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::LINE)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::FRONT)
        // .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
    let multisampling = vk::PipelineMultisampleStateCreateInfoBuilder::new()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlagBits::_1);
    let color_blend_attachments = vec![vk::PipelineColorBlendAttachmentStateBuilder::new()
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        )
        .blend_enable(false)];
    let color_blending = vk::PipelineColorBlendStateCreateInfoBuilder::new()
        .logic_op_enable(false)
        .attachments(&color_blend_attachments);
    let pipeline_stencil_info = vk::PipelineDepthStencilStateCreateInfoBuilder::new()
        .depth_test_enable(false)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .front(vk::StencilOpStateBuilder::new().build())
        .back(vk::StencilOpStateBuilder::new().build());
    let desc_layouts_slc = &[descriptor_set_layout];
    let push_constant_range = vk::PushConstantRangeBuilder::new()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(std::mem::size_of::<glm::Mat4>() as u32);
    let slice = [push_constant_range];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfoBuilder::new()
    .set_layouts(desc_layouts_slc)
    .push_constant_ranges(&slice);

    let pipeline_layout = Arc::new(
        device.create_pipeline_layout(&pipeline_layout_info, None).unwrap()
    );
    
    
    // let pipeline_layout_2 = device.create_pipeline_layout(&pipeline_layout_info, None).unwrap();
    let pipeline_info = vk::GraphicsPipelineCreateInfoBuilder::new()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .depth_stencil_state(&pipeline_stencil_info)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .color_blend_state(&color_blending)
        .layout(*pipeline_layout)
        .render_pass(*render_pass)
        .subpass(0);
    let pipeline = Arc::new(
        device.create_graphics_pipelines(vk::PipelineCache::default(), &[pipeline_info], None).unwrap()[0]
    );
    



    // let swapchain_framebuffers_pre: Vec<_> = swapchain_image_views
    //     .iter()
    //     .map(|image_view| {
    //         let attachments = vec![*image_view, depth_image_view];
    //         let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
    //             .render_pass(*render_pass)
    //             .attachments(&attachments)
    //             .width(swapchain_image_extent.width)
    //             .height(swapchain_image_extent.height)
    //             .layers(1);
    //         device.create_framebuffer(&framebuffer_info, None).unwrap()
    //     })
    //     .collect();

    


    // Now we have the renderpass/pipeline.

    // Next in 8700 is swapchain framebuffers creation.  Iirc 
    // these are created every frame, these are the first
    // precursor to creating command buffers.

    // The framebuffers and then 
    // the command buffer alloc.

    // Then we set up initial state, initial semaphores and are ready 
    // to create the render loop closure.

    let clear_values = vec![
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];


    let semaphore_info = vk::SemaphoreCreateInfoBuilder::new();
    let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);


    // let frame_resources = FrameResources {

    // }


    let frame_resources = FrameResources {

        device: device.clone(),
        swapchain: swapchain.clone(),
        swapchain_image_views: swapchain_image_views.clone(),
        depth_image_view: depth_image_view.clone(),
        swapchain_image_extent: swapchain_image_extent.clone(),
        render_pass: render_pass.clone(),
        queue_family: queue_family.clone(),
        clear_values: clear_values,
        pipeline: pipeline.clone(),
        pipeline_layout: pipeline_layout.clone(),
        index_buffer: index_buffer.clone(),
        vertex_buffer: vertex_buffer.clone(),
        descriptor_sets: descriptor_sets,
        indices_length: indices_length,
                // name: vec![0, 3, 7],
        ias: [ // image available semaphores
            device.create_semaphore(&semaphore_info, None).unwrap(),
            device.create_semaphore(&semaphore_info, None).unwrap(),
            device.create_semaphore(&semaphore_info, None).unwrap(),
        ],
        rfs: [ // render finished semaphores
            device.create_semaphore(&semaphore_info, None).unwrap(),
            device.create_semaphore(&semaphore_info, None).unwrap(),
            device.create_semaphore(&semaphore_info, None).unwrap(),
        ],
        iff: [ // in flight fences
            device.create_fence(&fence_info, None).unwrap(),
            device.create_fence(&fence_info, None).unwrap(),
            device.create_fence(&fence_info, None).unwrap(),
        ],
    };


    #[allow(clippy::collapsible_match, clippy::single_match)]
    event_loop.run(move |event, _, control_flow| match event {
        Event::NewEvents(StartCause::Init) => {
            *control_flow = ControlFlow::Poll;
        }
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            _ => (),
        },
        Event::DeviceEvent { event, .. } => match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(keycode),
                state,
                ..
            }) => match (keycode, state) {
                (VirtualKeyCode::Escape, ElementState::Released) => {
                    *control_flow = ControlFlow::Exit
                },
                // (winit::event::VirtualKeyCode::Space, ElementState::Released) => {  
                // },
                // (winit::event::VirtualKeyCode::Right, ElementState::Pressed) => {
                //     tx.send(0).unwrap();
                // },
                // (winit::event::VirtualKeyCode::Left, ElementState::Pressed) => {
                //     tx.send(1).unwrap();
                // },
                // (winit::event::VirtualKeyCode::Up, ElementState::Pressed) => {
                //     tx.send(2).unwrap();
                // },
                // (winit::event::VirtualKeyCode::Down, ElementState::Pressed) => {
                //     tx.send(3).unwrap();
                // },
                // (winit::event::VirtualKeyCode::Semicolon, ElementState::Pressed) => {
                //     tx.send(4).unwrap();
                // },
                // (winit::event::VirtualKeyCode::Q, ElementState::Pressed) => {
                //     tx.send(5).unwrap();
                // },
                _ => (),
            },
            _ => (),
        },
        Event::MainEventsCleared => {

            // in flight fences by frame which is a mutable variable
            // maybe in the multi-threaded context will be shared behind 
            // an arc and mutex.




            // device.wait_for_fences(&[in_flight_fences[frame]], true, u64::MAX).unwrap();
        
            // let image_index = device.acquire_next_image_khr
            // (
            //     swapchain,
            //     u64::MAX,
            //     // this semaphore will come from a set of frame resources
            //     // there will be three per thread.  so it won't be in an array
            //     // there will be one each of ias, rfs, and iff per set of frame resources
            //     image_available_semaphores[frame],
            //     vk::Fence::null(),
            // ).unwrap();

            // let image_in_flight = images_in_flight[image_index as usize];
            // if !image_in_flight.is_null() {
            //     device.lock().unwrap().wait_for_fences(&[image_in_flight], true, u64::MAX).unwrap();
            // }
            // images_in_flight[image_index as usize] = in_flight_fences[frame];
            // let wait_semaphores = vec![image_available_semaphores[frame]];
            // let framebuffer = swapchain_framebuffers_2.lock().unwrap()[image_index as usize];
            // let signal_semaphores = vec![render_finished_semaphores[frame]];
            
            // let cbs_35 = [*cursor_cb.lock().unwrap()];

            // let submit_info = vk::SubmitInfoBuilder::new()
            //     .wait_semaphores(&wait_semaphores)
            //     .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            //     .command_buffers(&cbs_35)
            //     .signal_semaphores(&signal_semaphores);
            
            // free_index = (image_index + 1) % 3;
            // rcb_tx.send(free_index);

            // let in_flight_fence = in_flight_fences[frame];
            // device.lock().unwrap().reset_fences(&[in_flight_fence]).unwrap();
            // device.lock().unwrap()
            //     .queue_submit(queue, &[submit_info], in_flight_fence)
            //     .unwrap();
            // let swapchains = vec![swapchain];
            // let image_indices = vec![image_index];
            // let present_info = vk::PresentInfoKHRBuilder::new()
            //     .wait_semaphores(&signal_semaphores)
            //     .swapchains(&swapchains)
            //     .image_indices(&image_indices);
            // device.lock().unwrap().queue_present_khr(queue, &present_info).unwrap();
            // frame = (frame + 1) % FRAMES_IN_FLIGHT;

        }

        Event::LoopDestroyed => unsafe {
            // println!("\n\n\n Loop destroyed, ending program.\n\n");
            // device.device_wait_idle().unwrap();
            // for &semaphore in image_available_semaphores
            //     .iter()
            //     .chain(render_finished_semaphores.iter())
            // {
            //     device.destroy_semaphore(semaphore, None);
            // }
            // for &fence in &in_flight_fences {
            //     device.destroy_fence(fence, None);
            // }
            // device.destroy_command_pool(command_pool, None);
            // for &framebuffer in &swapchain_framebuffers {
            //     device.destroy_framebuffer(framebuffer, None);
            // }
            // device.destroy_pipeline(pipeline, None);
            // device.destroy_render_pass(*render_pass, None);
            // device.destroy_pipeline_layout(pipeline_layout, None);
            // device.destroy_shader_module(shader_vert, None);
            // device.destroy_shader_module(shader_frag, None);
            // for &image_view in &swapchain_image_views {
            //     device.destroy_image_view(image_view, None);
            // }
            // device.destroy_swapchain_khr(swapchain, None);
            // device.destroy_device(None);
            // instance.destroy_surface_khr(surface, None);
            // // instance.destroy_debug_utils_messenger_ext(messenger, None);
            // // if !messenger.is_null() {
            // //     instance.destroy_debug_utils_messenger_ext(messenger, None);
            // // }
            // instance.destroy_instance(None);
            // println!("Exited cleanly");
        },
        _ => (),
    })


}



// per thread
// We could have 3 of these per thread.  Instead of 1 per thread with the arrays.
// First we'll do the single threaded version
// like this with array of three to index by frame 
// acquired from khr.  not sure about the transition 
// to multi-threaded, will reread that documentation material 
struct FrameResources {


    device: Arc<DeviceLoader>,
    
    swapchain: Arc<Mutex<SwapchainKHR>>,
    swapchain_image_views: Arc<Mutex<Vec<Arc<vk::ImageView>>>>,



    depth_image_view: Arc<vk::ImageView>,
    swapchain_image_extent: Arc<vk::Extent2D>,
    render_pass: Arc<vk::RenderPass>,
    queue_family: u32,
    clear_values: Vec<vk::ClearValue>,
    pipeline: Arc<vk::Pipeline>,
    pipeline_layout: Arc<vk::PipelineLayout>,
    index_buffer: Arc<vk::Buffer>,
    vertex_buffer: Arc<vk::Buffer>,
    descriptor_sets: SmallVec<vk::DescriptorSet>,
    indices_length: Arc<usize>,



    // frame: usize,

    
    // name: Vec<u32>,

    // images_in_flight: Vec<_>,


    ias: [vk::Semaphore; 3], // image available semaphores
    rfs: [vk::Semaphore; 3], // render finished semaphores
    iff: [vk::Fence; 3],
    // command_pools: [vk::CommandPool; 3],

}


struct FrameResources2 {
    ias: vk::Semaphore,
    rfs: vk::Semaphore,
    iff: vk::Fence,
    command_pool: vk::CommandPool,
    // pipeline: Arc pipeline
    // other shared stuff like the vertex buffers?
}


// Bettter Mutex usage with:
// pub fn get_mut(&mut self) -> LockResult<&mut T>


// pub fn into_inner(self) -> LockResult<T>
// where
// T: Sized, 
// Consumes this mutex, returning the underlying data.


unsafe fn draw_op_111
(
    device: Arc<DeviceLoader>,
    frame: usize,
    image_in_flight: vk::Fence,
    swapchain_image_views: Arc<Mutex<Vec<Arc<vk::ImageView>>>>,
    depth_image_view: Arc<vk::ImageView>,
    swapchain_image_extent: Arc<vk::Extent2D>,
    render_pass: Arc<vk::RenderPass>,
    queue_family: u32,
    clear_values: Vec<vk::ClearValue>,
    pipeline: Arc<vk::Pipeline>,
    pipeline_layout: Arc<vk::PipelineLayout>,
    index_buffer: Arc<vk::Buffer>,
    vertex_buffer: Arc<vk::Buffer>,
    descriptor_sets: SmallVec<vk::DescriptorSet>,
    indices_length: Arc<u32>,
)
{
    let image_view = swapchain_image_views.lock().unwrap()[frame].clone();   
    let attachments = vec![*image_view, *depth_image_view];
    let framebuffer_info = vk::FramebufferCreateInfoBuilder::new()
        .render_pass(*render_pass)
        .attachments(attachments.as_slice())
        .width(swapchain_image_extent.width)
        .height(swapchain_image_extent.height)
        .layers(1);
    let swapchain_framebuffer = device.create_framebuffer(&framebuffer_info, None).unwrap();
    // make new command pool?
    // I seem to recall that being a thing.
    // Iirc we make fresh swapchain framebuffer every frame.
    // Then allocate command buffers and record command buffer.
    // Now allocating and recording a command buffer.
    let info = vk::CommandPoolCreateInfoBuilder::new()
        .queue_family_index(queue_family)
        .flags(vk::CommandPoolCreateFlags::PROTECTED);
    let command_pool = device.create_command_pool(&info, None).unwrap();
    let info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = device.allocate_command_buffers(&info).unwrap()[0];
    let info = vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(command_buffer, &info).unwrap();
    let info = vk::RenderPassBeginInfoBuilder::new()
        .render_pass(*render_pass)
        .framebuffer(swapchain_framebuffer)
        .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: *swapchain_image_extent,
        })
        .clear_values(&clear_values);
    device.cmd_begin_render_pass(
        command_buffer,
        &info,
        vk::SubpassContents::INLINE,
    );
    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline);
    device.cmd_bind_index_buffer(command_buffer, *index_buffer, 0, vk::IndexType::UINT32);
    device.cmd_bind_vertex_buffers(command_buffer, 0, &[*vertex_buffer], &[0]);
    device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipeline_layout, 0, &descriptor_sets, &[]);


    let pc_view = glm::Mat4::identity();

    let ptr = std::ptr::addr_of!(pc_view) as *const c_void;
    device.cmd_push_constants
    (
        command_buffer,
        *pipeline_layout,
        vk::ShaderStageFlags::VERTEX,
        0,
        std::mem::size_of::<glm::Mat4>() as u32,
        ptr,
    );
    device.cmd_draw_indexed(command_buffer, *indices_length, *indices_length / 3, 0, 0, 0);
    device.cmd_end_render_pass(command_buffer);
    device.end_command_buffer(command_buffer).unwrap();
    // Ends recording of cb, but it has not been submitted yet.




}




unsafe fn create_swapchain
(
    device: Arc<DeviceLoader>,
    instance: Arc<InstanceLoader>,
    present_mode: Arc<PresentModeKHR>,
    format: Arc<SurfaceFormatKHR>,
    surface: Arc<SurfaceKHR>,
    physical_device: Arc<vk::PhysicalDevice>,
    window: Arc<Window>,
)
//  erupt::extensions::khr_swapchain::SwapchainKHR
-> Result<
    (
    Arc<Mutex<SwapchainKHR>>, 
    Arc<SmallVec<Image>>, 
    Arc<Mutex<Vec<Arc<vk::ImageView>>>>,
    Arc<vk::Extent2D>,
    ), String>
{
    let surface_caps = instance.get_physical_device_surface_capabilities_khr(*physical_device, *surface).unwrap();
    let mut image_count = surface_caps.min_image_count + 1;
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }
    // this is the stuff to recompute on window resize.
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
        .surface(*surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(swapchain_image_extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagBitsKHR::OPAQUE_KHR)
        .present_mode(*present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::default());
    let swapchain = device.create_swapchain_khr(&swapchain_info, None).unwrap();

    let swapchain_images = device.get_swapchain_images_khr(swapchain, None).unwrap();
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
            Arc::new(
                device.create_image_view(&image_view_info, None).unwrap()
            )
        })
        .collect();
    Ok((
        Arc::new(Mutex::new(swapchain)), 
        Arc::new(swapchain_images), 
        Arc::new(Mutex::new(swapchain_image_views)),
        Arc::new(swapchain_image_extent),
    ))
}

unsafe fn create_buffer
(
    device: Arc<DeviceLoader>,
    // flags: vk::BufferCreateFlags,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_type_index: u32,
    // queue_family_indices: &[u32],
) 
-> (vk::Buffer, vk::DeviceMemory) 
{
    let buffer_create_info = vk::BufferCreateInfoBuilder::new()
        // .flags(&[])
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(&[0]);
    let buffer = device.create_buffer(&buffer_create_info, None)
        .expect("Failed to create buffer.");
    let mem_reqs = device.get_buffer_memory_requirements(buffer);
    let allocate_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(memory_type_index);
    let buffer_memory = device
        .allocate_memory(&allocate_info, None)
        .expect("Failed to allocate memory for buffer.");
    device.bind_buffer_memory(buffer, buffer_memory, 0)
        .expect("Failed to bind buffer.");
    (buffer, buffer_memory)
}




unsafe fn buffer_vertices
(
    d: Arc<DeviceLoader>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    vertices: &mut Vec<VertexV3>,
)
-> Result<vk::Buffer, String>
{
    // let mut d = device;
    let vb_size = ((::std::mem::size_of_val(&(3.14 as f32))) * 9 * vertices.len()) as vk::DeviceSize;
    let info = vk::BufferCreateInfoBuilder::new()
        .size(vb_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let sb = d.create_buffer(&info, None).expect("Buffer create fail.");
    let mem_reqs = d.get_buffer_memory_requirements(sb);
    let info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(2);
    let sb_mem = d.allocate_memory(&info, None).unwrap();
    d.bind_buffer_memory(sb, sb_mem, 0).expect("Bind memory fail.");
    let data_ptr = d.map_memory(
        sb_mem,
        0,
        vk::WHOLE_SIZE,
        vk::MemoryMapFlags::empty(),
    ).unwrap() as *mut VertexV3;
    data_ptr.copy_from_nonoverlapping(vertices.as_ptr(), vertices.len());
    d.unmap_memory(sb_mem);
    let info = vk::BufferCreateInfoBuilder::new()
        .size(vb_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let vb = d.create_buffer(&info, None).expect("Create buffer fail.");
    let mem_reqs = d.get_buffer_memory_requirements(vb);
    let info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(1);
    let vb_mem = d.allocate_memory(&info, None).unwrap();
    d.bind_buffer_memory(vb, vb_mem, 0).expect("Bind memory fail.");
    let info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = d.allocate_command_buffers(&info).unwrap()[0];
    let info =  vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    d.begin_command_buffer(cb, &info).expect("Begin command buffer fail.");
    let info = vk::BufferCopyBuilder::new()
        .src_offset(0)
        .dst_offset(0)
        .size(vb_size);
    d.cmd_copy_buffer(cb, sb, vb, &[info]);
    d.end_command_buffer(cb).expect("End command buffer fail.");
    let cbs = &[cb];
    let info = vk::SubmitInfoBuilder::new()
        .wait_semaphores(&[])
        .command_buffers(cbs)
        .signal_semaphores(&[]);
    d.queue_submit(queue, &[info], vk::Fence::default()).expect("Queue submit fail.");
    Ok(vb)
}


pub unsafe fn buffer_indices
(
    device: Arc<DeviceLoader>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    indices: &mut Vec<u32>,
)
-> Result<(vk::Buffer), String>
{
    let ib_size = (::std::mem::size_of_val(&(10 as u32)) * indices.len()) as vk::DeviceSize;
    let info = vk::BufferCreateInfoBuilder::new()
        .size(ib_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let sb = device.create_buffer(&info, None).expect("Failed to create a staging buffer.");
    let mem_reqs = device.get_buffer_memory_requirements(sb);
    let info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(2);
    let sb_mem = device.allocate_memory(&info, None).unwrap();
    device.bind_buffer_memory(sb, sb_mem, 0).unwrap();
    let data_ptr = device.map_memory(
        sb_mem,
        0,
        vk::WHOLE_SIZE,
        vk::MemoryMapFlags::empty(),
    ).unwrap() as *mut u32;
    data_ptr.copy_from_nonoverlapping(indices.as_ptr(), indices.len());
    device.unmap_memory(sb_mem);
    // Todo: add destruction if this is still working
    let info = vk::BufferCreateInfoBuilder::new()
        .size(ib_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let ib = device.create_buffer(&info, None)
        .expect("Failed to create index buffer.");
    let mem_reqs = device.get_buffer_memory_requirements(ib);
    let alloc_info = vk::MemoryAllocateInfoBuilder::new()
        .allocation_size(mem_reqs.size)
        .memory_type_index(1);
    let ib_mem = device.allocate_memory(&alloc_info, None).unwrap();
    device.bind_buffer_memory(ib, ib_mem, 0);
    let info = vk::CommandBufferAllocateInfoBuilder::new()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cb = device.allocate_command_buffers(&info).unwrap()[0];
    let info =  vk::CommandBufferBeginInfoBuilder::new()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device.begin_command_buffer(cb, &info).expect("Failed begin_command_buffer.");
    let info =  vk::BufferCopyBuilder::new()
        .src_offset(0)
        .dst_offset(0)
        .size(ib_size);
    device.cmd_copy_buffer(cb, sb, ib, &[info]);
    let cbs = &[cb];
    device.end_command_buffer(cb).expect("Failed to end command buffer.");
    let info = vk::SubmitInfoBuilder::new()
        .wait_semaphores(&[])
        .command_buffers(cbs)
        .signal_semaphores(&[]);
    device.queue_submit(queue, &[info], vk::Fence::default()).expect("Failed to queue submit.");
    Ok(ib)
}

fn load_model
()
-> Result<(Vec<VertexV3>, Vec<u32>), String>
{
    let path_str: &str = "assets/terrain__002__.obj";
    let (models, materials) = tobj::load_obj(&path_str, &tobj::LoadOptions::default()).expect("Failed to load model object!");
    let model = models[0].clone();
    let mut vertices_terr: Vec<VertexV3> = vec![];
    let mesh = model.mesh;
    let total_vertices_count = mesh.positions.len() / 3;
    for i in 0..total_vertices_count {
        let vertex = VertexV3 {
            pos: [
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
                1.0,
            ],
            color: [0.8, 0.20, 0.30, 0.40],
        };
        vertices_terr.push(vertex);
    };
    let mut indices_terr_full = mesh.indices.clone(); 
    let mut indices_terr = vec![];
    for i in 0..(indices_terr_full.len() / 2) {
        indices_terr.push(indices_terr_full[i]);
    }
    Ok((vertices_terr, indices_terr))
}
    
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VertexV3 {
    pub pos: [f32; 4],
    pub color: [f32; 4],
}


#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct UniformBufferObject {
    pub model: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}

