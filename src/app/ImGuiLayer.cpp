#include <donut/app/ImGuiLayer.h>
#include <backends/imgui_impl_glfw.h>

namespace donut::app {

namespace {

struct ImGui_ImplNVRHI_InitInfo {
    donut::engine::ShaderFactory *ShaderFactory;
    nvrhi::IDevice *Device;
    nvrhi::ICommandList *CommandList;
    int NumFramesInFlight;
    nvrhi::Format RTVFormat;
};

// 
// imgui_impl_nvrhi declarations
// 
bool ImGui_ImplNVRHI_Init(ImGui_ImplNVRHI_InitInfo *info);
void ImGui_ImplNVRHI_Shutdown();
void ImGui_ImplNVRHI_NewFrame();
void ImGui_ImplNVRHI_RenderDrawData(ImDrawData *draw_data, nvrhi::IFramebuffer *fb);

bool ImGui_ImplNVRHI_CreateDeviceObjects();
void ImGui_ImplNVRHI_InvalidateDeviceObjects();
void Imgui_ImplNVRHI_UpdateTexture(ImTextureData *tex);

// 
// imgui_impl_nvrhi implements
//

struct ImGui_ImplNVRHI_Texture {
    nvrhi::TextureHandle Texture;
    nvrhi::StagingTextureHandle UploadBuffer;
    nvrhi::BindingSetHandle BindingSet;
};

struct ImGui_ImplNVRHI_Data {
    ImGui_ImplNVRHI_InitInfo InitInfo;

    nvrhi::IDevice *Device;
    nvrhi::ICommandList *CommandList;

    nvrhi::GraphicsPipelineHandle PSO;
    nvrhi::BindingLayoutHandle BindingLayout;
    nvrhi::SamplerHandle FontSampler;

    nvrhi::BufferHandle  IndexBuffer;
    nvrhi::BufferHandle VertexBuffer;
    uint32_t IndexBufferSize;
    uint32_t VertexBufferSize;

    ImVector<ImDrawVert> VtxBufferCache;
    ImVector<ImDrawIdx> IdxBufferCache;

};

struct VERTEX_CONSTANT_BUFFER_NVRHI {
    float ProjectST[4];
};

ImGui_ImplNVRHI_Data *ImGui_ImplNVRHI_GetBackendData() {
    return ImGui::GetCurrentContext()
               ? (ImGui_ImplNVRHI_Data *)ImGui::GetIO().BackendRendererUserData
               : nullptr;
}

void ImGui_ImplNVRHI_DestroyTexture(ImTextureData *tex) {
    ImGui_ImplNVRHI_Texture *backend_tex = (ImGui_ImplNVRHI_Texture *)tex->BackendUserData;
    if(backend_tex == nullptr)
        return;

    backend_tex->Texture = nullptr;
    backend_tex->UploadBuffer = nullptr;
    backend_tex->BindingSet = nullptr;
    IM_DELETE(backend_tex);

    tex->SetTexID(ImTextureID_Invalid);
    tex->SetStatus(ImTextureStatus_Destroyed);
    tex->BackendUserData = nullptr;
}

bool ImGui_ImplNVRHI_CreateDeviceObjects() {
    auto bd = ImGui_ImplNVRHI_GetBackendData();

    if(bd->PSO)
        ImGui_ImplNVRHI_InvalidateDeviceObjects();

    // Create PSO
    {
        auto vs = bd->InitInfo.ShaderFactory->CreateShader(
            "donut/imgui_vertex.hlsl", "main", {}, nvrhi::ShaderType::Vertex);

        auto ps = bd->InitInfo.ShaderFactory->CreateShader(
            "donut/imgui_pixel", "main", {}, nvrhi::ShaderType::Pixel);

        if (!vs || !ps) return false;

        // create attribute layout object
        nvrhi::VertexAttributeDesc vertexAttribLayout[] = {
            {"POSITION", nvrhi::Format::RG32_FLOAT, 1, 0, offsetof(ImDrawVert, pos),
             sizeof(ImDrawVert), false},
            {"TEXCOORD", nvrhi::Format::RG32_FLOAT, 1, 0, offsetof(ImDrawVert, uv),
             sizeof(ImDrawVert), false},
            {"COLOR", nvrhi::Format::RGBA8_UNORM, 1, 0, offsetof(ImDrawVert, col),
             sizeof(ImDrawVert), false},
        };

        nvrhi::InputLayoutHandle shaderAttribLayout;
        bd->Device->createInputLayout(
            vertexAttribLayout, sizeof(vertexAttribLayout) / sizeof(vertexAttribLayout[0]),
            vs, &shaderAttribLayout);

        // create PSO
        {
            nvrhi::BlendState blendState;
            blendState.targets[0]
                .setBlendEnable(true)
                .setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
                .setDestBlend(nvrhi::BlendFactor::InvSrcAlpha)
                .setSrcBlendAlpha(nvrhi::BlendFactor::One)
                .setDestBlendAlpha(nvrhi::BlendFactor::InvSrcAlpha);

            auto rasterState = nvrhi::RasterState()
                                   .setFillSolid()
                                   .setCullNone()
                                   .setScissorEnable(true)
                                   .setDepthClipEnable(true);

            auto depthStencilState = nvrhi::DepthStencilState()
                                         .disableDepthTest()
                                         .enableDepthWrite()
                                         .disableStencil()
                                         .setDepthFunc(nvrhi::ComparisonFunc::Always);

            nvrhi::RenderState renderState;
            renderState.blendState = blendState;
            renderState.depthStencilState = depthStencilState;
            renderState.rasterState = rasterState;

            nvrhi::BindingLayoutDesc layoutDesc;
            layoutDesc.visibility = nvrhi::ShaderType::Vertex | nvrhi::ShaderType::Pixel;
            layoutDesc.bindings = {
                nvrhi::BindingLayoutItem::PushConstants(0, sizeof(VERTEX_CONSTANT_BUFFER_NVRHI)),
                nvrhi::BindingLayoutItem::Texture_SRV(0),
                nvrhi::BindingLayoutItem::Sampler(0)};
            bd->Device->createBindingLayout(layoutDesc, &bd->BindingLayout);

            nvrhi::GraphicsPipelineDesc basePSODesc;
            basePSODesc.primType = nvrhi::PrimitiveType::TriangleList;
            basePSODesc.inputLayout = shaderAttribLayout;
            basePSODesc.VS = vs;
            basePSODesc.PS = ps;
            basePSODesc.renderState = renderState;
            basePSODesc.bindingLayouts = {bd->BindingLayout};

            nvrhi::FramebufferInfo fbInfo;
            fbInfo.colorFormats = { bd->InitInfo.RTVFormat };

            bd->Device->createGraphicsPipeline1(basePSODesc, fbInfo, &bd->PSO);
        }
    }

    {
        const auto desc = nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setAllFilters(true);

        bd->Device->createSampler(desc, &bd->FontSampler);

        if (bd->FontSampler == nullptr)
            return false;
    }

    return true;
}

void ImGui_ImplNVRHI_InvalidateDeviceObjects() {
    auto bd = ImGui_ImplNVRHI_GetBackendData();
    if(!bd)
        return;

    bd->PSO = nullptr;
    bd->BindingLayout = nullptr;
    bd->FontSampler = nullptr;
    bd->VertexBuffer = nullptr;
    bd->IndexBuffer = nullptr;

    // Destroy all textures
    for(ImTextureData *tex : ImGui::GetPlatformIO().Textures) {
        if(tex->RefCount == 1)
            ImGui_ImplNVRHI_DestroyTexture(tex);
    }
}

void Imgui_ImplNVRHI_UpdateTexture(ImTextureData *tex) {
    auto bd = ImGui_ImplNVRHI_GetBackendData();

    if(tex->Status == ImTextureStatus_WantCreate) {
        IM_ASSERT(tex->TexID == ImTextureID_Invalid && tex->BackendUserData == nullptr);
        IM_ASSERT(tex->Format == ImTextureFormat_RGBA32);
        ImGui_ImplNVRHI_Texture *backend_tex = IM_NEW(ImGui_ImplNVRHI_Texture)();

        nvrhi::TextureDesc desc;
        desc.dimension = nvrhi::TextureDimension::Texture2D;
        desc.width = tex->Width;
        desc.height = tex->Height;
        desc.format = nvrhi::Format::RGBA8_UNORM;
        desc.initialState = nvrhi::ResourceStates::ShaderResource;
        desc.keepInitialState = true;
        desc.debugName = "ImGui texture";

        bd->Device->createTexture(desc, &backend_tex->Texture);

        nvrhi::BindingSetDesc bindingDesc;
        bindingDesc.bindings = {
            nvrhi::BindingSetItem::PushConstants(0, sizeof(VERTEX_CONSTANT_BUFFER_NVRHI)),
            nvrhi::BindingSetItem::Texture_SRV(0, backend_tex->Texture),
            nvrhi::BindingSetItem::Sampler(0, bd->FontSampler)
        };
        IM_ASSERT(bd->BindingLayout);
        bd->Device->createBindingSet(bindingDesc, bd->BindingLayout, &backend_tex->BindingSet);

        tex->SetTexID((ImTextureID)backend_tex->BindingSet.Get());
        tex->BackendUserData = backend_tex;
    }

    if(tex->Status == ImTextureStatus_WantUpdates || tex->Status == ImTextureStatus_WantCreate) {
        auto backend_tex = (ImGui_ImplNVRHI_Texture *)tex->BackendUserData;
        IM_ASSERT(tex->Format == ImTextureFormat_RGBA32);

        // We could use the smaller rect on _WantCreate but using the full rect allows us to clear the texture.
        // FIXME-OPT: Uploading single box even when using ImTextureStatus_WantUpdates. Could use tex->Updates[]
        // - Copy all blocks contiguously in upload buffer.
        // - Barrier before copy, submit all CopyTextureRegion(), barrier after copy.
        const uint32_t upload_x = (tex->Status == ImTextureStatus_WantCreate) ? 0 : tex->UpdateRect.x;
        const uint32_t upload_y = (tex->Status == ImTextureStatus_WantCreate) ? 0 : tex->UpdateRect.y;
        const uint32_t upload_w = (tex->Status == ImTextureStatus_WantCreate) ? tex->Width : tex->UpdateRect.w;
        const uint32_t upload_h =
            (tex->Status == ImTextureStatus_WantCreate) ? tex->Height : tex->UpdateRect.h;

        bool recreate_upload_buffer = !backend_tex->UploadBuffer;
        if(!recreate_upload_buffer) {
            auto &desc = backend_tex->UploadBuffer->getDesc();
            recreate_upload_buffer = desc.width < upload_w || desc.height < upload_h;
        }
        if(recreate_upload_buffer) {
            nvrhi::TextureDesc desc;
            desc.width = upload_w;
            desc.height = upload_h;
            desc.format = nvrhi::Format::RGBA8_UNORM;
            bd->Device->createStagingTexture(desc, nvrhi::CpuAccessMode::Write, &backend_tex->UploadBuffer);
        }

        size_t row_pitch;
        auto data = (uint8_t *)bd->Device->mapStagingTexture(
            backend_tex->UploadBuffer, {}, nvrhi::CpuAccessMode::Write, row_pitch);
        for (uint32_t y = 0; y < upload_h; ++y, data += row_pitch)
            memcpy(data, tex->GetPixelsAt(upload_x, y + upload_y), upload_w * tex->BytesPerPixel);
        bd->Device->unmapStagingTexture(backend_tex->UploadBuffer);

        bd->CommandList->copyTexture3(
            backend_tex->Texture,
            nvrhi::TextureSlice{upload_x, upload_y, 0, upload_w, upload_h, 1},
            backend_tex->UploadBuffer,
            nvrhi::TextureSlice{0, 0, 0, upload_w, upload_h, 1});

        tex->SetStatus(ImTextureStatus_OK);
    }

    if(tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames >= (int)bd->InitInfo.NumFramesInFlight)
        ImGui_ImplNVRHI_DestroyTexture(tex);
}

bool ImGui_ImplNVRHI_Init(ImGui_ImplNVRHI_InitInfo *init_info) {
    auto &io = ImGui::GetIO();
    IMGUI_CHECKVERSION();
    IM_ASSERT(io.BackendRendererUserData == nullptr && "Already initialized a renderer backend!");

    auto bd = IM_NEW(ImGui_ImplNVRHI_Data)();
    bd->InitInfo = *init_info;
    init_info = &bd->InitInfo;

    io.BackendRendererUserData = (void *)bd;
    io.BackendRendererName = "imgui_impl_nvrhi";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;

#if DONUT_WITH_DX11
    if(init_info->Device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D11) {
        auto &platform_io = ImGui::GetPlatformIO();
        platform_io.Renderer_TextureMaxWidth = platform_io.Renderer_TextureMaxHeight = D3D11_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION;
    }
#endif

    bd->Device = init_info->Device;
    bd->CommandList = init_info->CommandList;
    bd->VertexBufferSize = 0;
    bd->IndexBufferSize = 0;

    return true;
}

void ImGui_ImplNVRHI_Shutdown() {
    auto bd = ImGui_ImplNVRHI_GetBackendData();
    IM_ASSERT(bd != nullptr && "No renderer backend to shutdown, or already shutdown?");
    ImGuiIO& io = ImGui::GetIO();

    ImGui_ImplNVRHI_InvalidateDeviceObjects();

    io.BackendRendererName = nullptr;
    io.BackendRendererUserData = nullptr;
    io.BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset | ImGuiBackendFlags_RendererHasTextures);
    IM_DELETE(bd);
}

void ImGui_ImplNVRHI_NewFrame() {
    auto bd = ImGui_ImplNVRHI_GetBackendData();
    IM_ASSERT(bd != nullptr &&
              "Context or backend not initialized! Did you call ImGui_ImplDX12_Init()?");
    
    if(!bd->PSO)
        if(!ImGui_ImplNVRHI_CreateDeviceObjects())
            IM_ASSERT(0 && "ImGui_ImplNVRHI_CreateDeviceObjects() failed!");
}

void ImGui_ImplNVRHI_RenderDrawData(ImDrawData *draw_data, nvrhi::IFramebuffer *fb) {
    // Avoid rendering when minimized
    if (draw_data->DisplaySize.x <= 0.0f || draw_data->DisplaySize.y <= 0.0f) return;

    // Catch up with texture updates. Most of the times, the list will have 1 element with
    // an OK status, aka nothing to do. (This almost always points to
    // ImGui::GetPlatformIO().Textures[] but is part of ImDrawData to allow overriding or
    // disabling texture updates).
    if (draw_data->Textures != nullptr)
        for (ImTextureData *tex : *draw_data->Textures)
            if (tex->Status != ImTextureStatus_OK) Imgui_ImplNVRHI_UpdateTexture(tex);


    auto bd = ImGui_ImplNVRHI_GetBackendData();
    // Create and grow vertex/index buffers if needed

    if(bd->VertexBuffer == nullptr || bd->VertexBufferSize < (uint32_t)draw_data->TotalVtxCount) {
        bd->VertexBufferSize = draw_data->TotalVtxCount + 5000;
        nvrhi::BufferDesc desc;
        desc.byteSize = bd->VertexBufferSize * sizeof(ImDrawVert);
        desc.debugName = "ImGui vertex buffer";
        desc.isVertexBuffer = true;
        desc.initialState = nvrhi::ResourceStates::VertexBuffer;
        desc.keepInitialState = true;

        bd->Device->createBuffer(desc, &bd->VertexBuffer);
        bd->VtxBufferCache.resize(bd->VertexBufferSize);
    }
    if(bd->IndexBuffer == nullptr || bd->IndexBufferSize < (uint32_t)draw_data->TotalIdxCount) {
        bd->IndexBufferSize = draw_data->TotalIdxCount + 10000;
        nvrhi::BufferDesc desc;
        desc.byteSize = bd->IndexBufferSize * sizeof(ImDrawIdx);
        desc.debugName = "ImGui index buffer";
        desc.isIndexBuffer = true;
        desc.initialState = nvrhi::ResourceStates::IndexBuffer;
        desc.keepInitialState = true;

        bd->Device->createBuffer(desc, &bd->IndexBuffer);
        bd->IdxBufferCache.resize(bd->IndexBufferSize);
    }

    // Upload vertex/index data into a single contiguous GPU buffer
    ImDrawVert* vtx_dst = bd->VtxBufferCache.Data;
    ImDrawIdx* idx_dst = bd->IdxBufferCache.Data;
    for(int n = 0; n < draw_data->CmdListsCount; ++n) {
        const ImDrawList *draw_list = draw_data->CmdLists[n];
        memcpy(vtx_dst, draw_list->VtxBuffer.Data, draw_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idx_dst, draw_list->IdxBuffer.Data, draw_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtx_dst += draw_list->VtxBuffer.Size;
        idx_dst += draw_list->IdxBuffer.Size;
    }

    IM_ASSERT(vtx_dst- bd->VtxBufferCache.Data == draw_data->TotalVtxCount);
    IM_ASSERT(idx_dst - bd->IdxBufferCache.Data == draw_data->TotalIdxCount);
    if(draw_data->TotalVtxCount)
        bd->CommandList->writeBuffer(bd->VertexBuffer, bd->VtxBufferCache.Data,
                                     draw_data->TotalVtxCount * sizeof(ImDrawVert));
    if(draw_data->TotalIdxCount)
        bd->CommandList->writeBuffer(bd->IndexBuffer, bd->IdxBufferCache.Data,
                                     draw_data->TotalIdxCount * sizeof(ImDrawIdx));

    // Setup default render state
    nvrhi::GraphicsState state;
    VERTEX_CONSTANT_BUFFER_NVRHI vertex_constant_buffer;
    {
        state.framebuffer = fb;
        state.pipeline = bd->PSO;
        // Set default binding set
        auto &platform_io = ImGui::GetPlatformIO();
        auto defaultBinding = (nvrhi::IBindingSet *)platform_io.Textures[0]->GetTexID();
        state.bindings = {defaultBinding};

        nvrhi::Viewport vp {
            0, draw_data->DisplaySize.x * draw_data->FramebufferScale.x, 0,
            draw_data->DisplaySize.y * draw_data->FramebufferScale.y, 0.f, 1.f
        };
        state.viewport.addViewport(vp);
        state.viewport.scissorRects.resize(1);
        state.viewport.scissorRects[0] =
            nvrhi::Rect{(int)vp.minX, (int)vp.maxX, (int)vp.minY, (int)vp.maxY};

        nvrhi::VertexBufferBinding vbufBinding;
        vbufBinding.buffer = bd->VertexBuffer;
        vbufBinding.slot = 0;
        vbufBinding.offset = 0;
        state.addVertexBuffer(vbufBinding);
        state.indexBuffer.setBuffer(bd->IndexBuffer)
            .setOffset(0)
            .setFormat(sizeof(ImDrawIdx) == 2 ? nvrhi::Format::R16_UINT
                                              : nvrhi::Format::R32_UINT);

        bd->CommandList->setGraphicsState(state);

        float L = draw_data->DisplayPos.x;
        float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
        float T = draw_data->DisplayPos.y;
        float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
        vertex_constant_buffer.ProjectST[0] = 2.f / (R - L);
        vertex_constant_buffer.ProjectST[1] = 2.f / (T - B);
        vertex_constant_buffer.ProjectST[2] = (R + L) / (L - R);
        vertex_constant_buffer.ProjectST[3] = (T + B) / (B - T);

        bd->CommandList->setPushConstants(&vertex_constant_buffer, sizeof(vertex_constant_buffer));
    }

     // Render command lists
    // (Because we merged all buffers into a single one, we maintain our own offset into them)
    int vtx_offset = 0;
    int idx_offset = 0;
    ImVec2 clip_off = draw_data->DisplayPos;
    ImVec2 clip_scale = draw_data->FramebufferScale;
    for(int n = 0; n < draw_data->CmdListsCount; ++n) {
        const ImDrawList *draw_list = draw_data->CmdLists[n];
        for(int i = 0; i < draw_list->CmdBuffer.Size; ++i) {
            const ImDrawCmd *draw_cmd = &draw_list->CmdBuffer[i];
            if(draw_cmd->UserCallback) {
                // User callback, registered via ImDrawList::AddCallback()
                // (ImDrawCallback_ResetRenderState is a special callback value used by the
                // user to request the renderer to reset render state.)
                if(draw_cmd->UserCallback == ImDrawCallback_ResetRenderState) {
                    bd->CommandList->setGraphicsState(state);
                    bd->CommandList->setPushConstants(&vertex_constant_buffer,
                                                      sizeof(vertex_constant_buffer));
                } else
                    draw_cmd->UserCallback(draw_list, draw_cmd);
            } else {
                // Project scissor/clipping rectangles into framebuffer space
                ImVec2 clip_min((draw_cmd->ClipRect.x - clip_off.x) * clip_scale.x,
                                (draw_cmd->ClipRect.y - clip_off.y) * clip_scale.y);
                ImVec2 clip_max((draw_cmd->ClipRect.z - clip_off.x) * clip_scale.x,
                                (draw_cmd->ClipRect.w - clip_off.y) * clip_scale.y);
                // Test the integer rect: a sub-pixel clip truncates to an empty scissor.
                if ((int)clip_max.x <= (int)clip_min.x || (int)clip_max.y <= (int)clip_min.y) continue;

                // Apply scissor/clipping rectangle
                state.viewport.scissorRects[0] = nvrhi::Rect{
                    (int)clip_min.x, (int)clip_max.x, (int)clip_min.y, (int)clip_max.y};

                auto binding = (nvrhi::IBindingSet *)draw_cmd->GetTexID();
                state.bindings = {binding};
                bd->CommandList->setGraphicsState(state);

                bd->CommandList->setPushConstants(&vertex_constant_buffer,
                                                  sizeof(vertex_constant_buffer));

                nvrhi::DrawArguments args;
                args.startIndexLocation = draw_cmd->IdxOffset + idx_offset;
                args.startVertexLocation = draw_cmd->VtxOffset + vtx_offset;
                args.vertexCount = draw_cmd->ElemCount;
                bd->CommandList->drawIndexed(args);
            }
        }
        idx_offset += draw_list->IdxBuffer.Size;
        vtx_offset += draw_list->VtxBuffer.Size;
    }
}

}

ImGuiLayer::ImGuiLayer(DeviceManager *deviceManager): IAppLayer(deviceManager) {
}

ImGuiLayer::~ImGuiLayer() {
    ImGui_ImplNVRHI_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(m_imContext);
}

bool ImGuiLayer::Init(engine::ShaderFactory *pShaderFactory) {

    m_imContext = ImGui::CreateContext();

    GetDevice()->createCommandList(nvrhi::CommandListParameters(), &m_commandList);

    m_shaderFactory = pShaderFactory;

    ImGui_ImplNVRHI_InitInfo initInfo;
    initInfo.Device = GetDevice();
    initInfo.CommandList = m_commandList;
    initInfo.NumFramesInFlight = GetDeviceManager()->GetBackBufferCount();
    initInfo.ShaderFactory = m_shaderFactory;
    initInfo.RTVFormat =
        GetDeviceManager()->GetFramebuffer(0)->getFramebufferInfo().colorFormats[0];

    ImGui_ImplNVRHI_Init(&initInfo);
    ImGui_ImplGlfw_InitForOther(GetDeviceManager()->GetWindow(), false);

    return true;
}

ImGuiContext *ImGuiLayer::GetImContext() { return m_imContext; }

bool ImGuiLayer::KeyboardUpdate(int key, int scancode, int action, int mods) {

    ImGui_ImplGlfw_KeyCallback(GetDeviceManager()->GetWindow(), key, scancode, action, mods);
    return ImGui::GetIO().WantCaptureKeyboard;
}

// void ImGuiLayer::WindowFocused(int focused) {
//     ImGui_ImplGlfw_WindowFocusCallback(GetDeviceManager()->GetWindow(), focused);
// }

bool ImGuiLayer::KeyboardCharInput(unsigned int unicode, int mods) {
    ImGui_ImplGlfw_CharCallback(GetDeviceManager()->GetWindow(), unicode);
    return ImGui::GetIO().WantCaptureKeyboard;
}

// Workaround: X11 seems to send spurious Leave/Enter events which would make us lose our position,
// so we back it up and restore on Leave/Enter (see https://github.com/ocornut/imgui/issues/4984)
// bool ImGuiLayer::CursorEnter(int entered) {
//     ImGui_ImplGlfw_CursorEnterCallback(GetDeviceManager()->GetWindow(), entered);
//     return false;
// }

bool ImGuiLayer::MousePosUpdate(double xpos, double ypos) {
    ImGui_ImplGlfw_CursorPosCallback(GetDeviceManager()->GetWindow(), xpos, ypos);
    auto &io = ImGui::GetIO();
    return io.WantCaptureMouse;
}

bool ImGuiLayer::MouseScrollUpdate(double xoffset, double yoffset) {
    ImGui_ImplGlfw_ScrollCallback(GetDeviceManager()->GetWindow(), xoffset, yoffset);
    auto &io = ImGui::GetIO();
    return io.WantCaptureMouse;
}

bool ImGuiLayer::MouseButtonUpdate(int button, int action, int mods) {
    ImGui_ImplGlfw_MouseButtonCallback(GetDeviceManager()->GetWindow(), button, action, mods);
    auto &io = ImGui::GetIO();
    return io.WantCaptureMouse;
}

void ImGuiLayer::Animate(float elapsedTimeSeconds) {
    ImGui_ImplNVRHI_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();
}

void ImGuiLayer::Render(nvrhi::IFramebuffer *framebuffer) {

    BuildUI();

    ImGui::Render();

    m_commandList->open();
    ImGui_ImplNVRHI_RenderDrawData(ImGui::GetDrawData(), framebuffer);

    m_commandList->close();
    GetDevice()->executeCommandList(m_commandList);
}

void ImGuiLayer::DisplayScaleChanged(float scaleX, float scaleY) {
    // Clear the ImGui font atlas and invalidate the font texture
    // to re-register and re-rasterize all fonts on the next frame (see Animate)
}

void ImGuiLayer::BeginFullScreenWindow() {
    ImGuiIO const &io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(0.f, 0.f), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x / io.DisplayFramebufferScale.x,
                                    io.DisplaySize.y / io.DisplayFramebufferScale.y),
                             ImGuiCond_Always);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.f);
    ImGui::SetNextWindowBgAlpha(0.f);
    ImGui::Begin(" ", 0,
                 ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoScrollbar);
}

void ImGuiLayer::DrawScreenCenteredText(const char *text) {
    ImGuiIO const &io = ImGui::GetIO();
    ImVec2 textSize = ImGui::CalcTextSize(text);
    ImGui::SetCursorPosX((io.DisplaySize.x / io.DisplayFramebufferScale.x - textSize.x) *
                         0.5f);
    ImGui::SetCursorPosY((io.DisplaySize.y / io.DisplayFramebufferScale.y - textSize.y) *
                         0.5f);
    ImGui::TextUnformatted(text);
}

void ImGuiLayer::EndFullScreenWindow() {
    ImGui::End();
    ImGui::PopStyleVar();
}

}  // namespace donut