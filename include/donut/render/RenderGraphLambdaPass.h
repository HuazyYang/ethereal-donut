/*
* Copyright (c) 2026, ethereal contributors. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// One-off graph passes from lambdas (design 12.1). Header-only and outside the ABI surface: the pass object's
// virtual table lives in the caller's module.
//
//     const auto& data = AddLambdaPass<BlurData>(graph, { "Blur", PassType::Compute, PassFlags::None },
//         [&](ISetupContext* setup, BlurData& d) { d.output = setup->Write(output, Access::UAVWrite, ...); },
//         [=](IPassContext* context, const BlurData& d) { ... });
//
// The pass object and its Data live in the frame arena. The graph releases its reference at Reset; the object's
// destruction then runs the destructors of the lambdas and of Data but frees nothing, because the arena owns the
// memory. It is the one arena object with a destructor, so the lambdas' captures may own resources. Data itself
// follows the frame-data rule: trivially destructible.

#pragma once

#include <donut/render/RenderGraph.h>
#include <nvrhi/core/foundation.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace donut::render
{
    namespace detail
    {
        // MakeNewRCObj's allocator: memory from the frame arena, released with it.
        struct FrameArenaAllocator
        {
            IRenderGraph* graph;

            void* Allocate(size_t size) noexcept { return graph->AllocateFrameData(size, alignof(std::max_align_t)); }
            void Free(void*) noexcept { }
        };

        struct NoPrepare
        {
            template <typename Data> void operator()(IPrepareContext*, Data&) const noexcept { }
        };

        template <typename Data, typename SetupFn, typename PrepareFn, typename ExecFn>
        class LambdaPass final : public nvrhi::ObjectImpl<IGpuPass>
        {
        public:
            NVRHI_BEGIN_INTERFACE_TABLE_INLINE(LambdaPass)
            NVRHI_IMPLEMENTS_INTERFACE(IGpuPass)
            NVRHI_END_INTERFACE_TABLE()

            LambdaPass(SetupFn&& setup, PrepareFn&& prepare, ExecFn&& exec)
                : m_Setup(std::forward<SetupFn>(setup))
                , m_Prepare(std::forward<PrepareFn>(prepare))
                , m_Exec(std::forward<ExecFn>(exec))
            { }

            nvrhi::FRESULT Setup(ISetupContext* context, void*) noexcept override
            {
                m_Setup(context, m_Data);
                return nvrhi::FS_OK;
            }

            nvrhi::FRESULT Prepare(IPrepareContext* context, void*) noexcept override
            {
                m_Prepare(context, m_Data);
                return nvrhi::FS_OK;
            }

            nvrhi::FRESULT Execute(IPassContext* context, const void*) noexcept override
            {
                m_Exec(context, static_cast<const Data&>(m_Data));
                return nvrhi::FS_OK;
            }

            void OnResourcesReleased() noexcept override { }

            Data m_Data{};

        private:
            std::decay_t<SetupFn> m_Setup;
            std::decay_t<PrepareFn> m_Prepare;
            std::decay_t<ExecFn> m_Exec;
        };

        template <typename Data, typename SetupFn, typename PrepareFn, typename ExecFn>
        const Data* AddLambdaPassImpl(IRenderGraph* graph, const PassDesc& desc, SetupFn&& setup, PrepareFn&& prepare,
                                      ExecFn&& exec, PassId* outPass)
        {
            static_assert(std::is_trivially_destructible_v<Data>,
                          "lambda pass data follows the frame-data rule: trivially destructible");
            using Pass = LambdaPass<Data, SetupFn, PrepareFn, ExecFn>;

            void* memory = graph->AllocateFrameData(sizeof(FrameArenaAllocator), alignof(FrameArenaAllocator));
            if (!memory)
                return nullptr;
            FrameArenaAllocator* allocator = new (memory) FrameArenaAllocator{ graph };
            Pass* pass = nvrhi::MakeNewRCObj<FrameArenaAllocator>(allocator).template RcNew<Pass>(
                std::forward<SetupFn>(setup), std::forward<PrepareFn>(prepare), std::forward<ExecFn>(exec));
            if (!pass)
                return nullptr;

            // AddPass takes its own reference; ours goes at once, so the graph owns the object for the frame.
            graph->AddPass(desc, pass, &pass->m_Data, outPass);
            const Data* data = &pass->m_Data;
            pass->Release();
            return data;
        }
    }

    // setup(ISetupContext*, Data&) declares the accesses; exec(IPassContext*, const Data&) records the work.
    // Returns the pass's Data, which Setup has filled, so that the caller can read the ids it wrote.
    template <typename Data, typename SetupFn, typename ExecFn>
    const Data& AddLambdaPass(IRenderGraph* graph, const PassDesc& desc, SetupFn&& setup, ExecFn&& exec,
                              _Out_opt_ PassId* outPass = nullptr)
    {
        static const Data s_Empty{};
        const Data* data = detail::AddLambdaPassImpl<Data>(graph, desc, std::forward<SetupFn>(setup),
            detail::NoPrepare{}, std::forward<ExecFn>(exec), outPass);
        return data ? *data : s_Empty;
    }

    // With prepare(IPrepareContext*, Data&), which runs serially before recording (13.1).
    template <typename Data, typename SetupFn, typename PrepareFn, typename ExecFn>
    const Data& AddLambdaPassWithPrepare(IRenderGraph* graph, const PassDesc& desc, SetupFn&& setup,
                                         PrepareFn&& prepare, ExecFn&& exec, _Out_opt_ PassId* outPass = nullptr)
    {
        static const Data s_Empty{};
        const Data* data = detail::AddLambdaPassImpl<Data>(graph, desc, std::forward<SetupFn>(setup),
            std::forward<PrepareFn>(prepare), std::forward<ExecFn>(exec), outPass);
        return data ? *data : s_Empty;
    }
}
