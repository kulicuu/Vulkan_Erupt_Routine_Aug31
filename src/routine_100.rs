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
        // .with_maximized(true)
        .with_inner_size(winit::dpi::LogicalSize::new(2000, 2000))
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
    // let device  = Arc::new(Mutex::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap()));
    let device  = Arc::new(DeviceLoader::new(&instance, physical_device, &device_info).unwrap());

    // let mut dee = device.lock().unwrap();
    let queue = device.get_device_queue(queue_family, 0);

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
            device.create_image_view(&image_view_info, None).unwrap()
        })
        .collect();
    
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


    let semaphore_info = vk::SemaphoreCreateInfoBuilder::new();
    let fence_info = vk::FenceCreateInfoBuilder::new().flags(vk::FenceCreateFlags::SIGNALED);
    let command_pool_info = vk::CommandPoolCreateInfoBuilder::new()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family);

    let frames_resources: Vec<FrameResources> = (0..3)
    .map(|_| {
        FrameResources {
            name: vec![0, 3, 7],
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
            command_pools: [
                device.create_command_pool(&command_pool_info, None).unwrap(),
                device.create_command_pool(&command_pool_info, None).unwrap(),
                device.create_command_pool(&command_pool_info, None).unwrap(),
            ]
        }
    })
    .collect();

    println!("frames_resources.len() {}", frames_resources.len());

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

    let physical_device_memory_properties = instance.get_physical_device_memory_properties(physical_device);

    let info = vk::CommandPoolCreateInfoBuilder::new()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let tcp = device.create_command_pool(&command_pool_info, None).unwrap();


    let vb = Arc::new(
        buffer_vertices(
            device.clone(),
            queue,
            tcp,
            &mut vertices_terr,
        ).unwrap()
    );

    let ib = Arc::new(
        buffer_indices(
            device.clone(),
            queue,
            tcp,
            &mut indices_terr,
        ).unwrap()
    );
    
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
    // I imagine that one needs to be shared among threads.
    let uniform_buffer = Arc::new(Mutex::new(
        uniform_buffer,
    ));
    let uniform_buffer_memory = Arc::new(Mutex::new(
        uniform_buffer_memory,
    ));
    
    



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
-> (vk::Buffer, vk::DeviceMemory) {
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


// per thread
struct FrameResources {
    name: Vec<u32>,
    ias: [vk::Semaphore; 3], // image available semaphores
    rfs: [vk::Semaphore; 3], // render finished semaphores
    iff: [vk::Fence; 3],
    command_pools: [vk::CommandPool; 3],
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

