#ifndef IMGUILAYER_H
#define IMGUILAYER_H
#include <donut/app/DeviceManager.h>
#include <donut/engine/ShaderFactory.h>
#include <imgui.h>

namespace donut::engine {
class ShaderFactory;
}

namespace donut::app {

class ImGuiLayer : public IAppLayer {
    NVRHI_INHERIT_INTERFACE_TABLE()
 public:
    ImGuiLayer(DeviceManager *deviceManager);
    ~ImGuiLayer();

    bool Init(engine::ShaderFactory *pShaderFactory);
    ImGuiContext *GetImContext();

   //  void WindowFocused(int focused) override;
    bool KeyboardUpdate(int key, int scancode, int action, int mods) override;
    bool KeyboardCharInput(unsigned int unicode, int mods) override;
   //  bool CursorEnter(int entered) override;
    bool MousePosUpdate(double xpos, double ypos) override;
    bool MouseScrollUpdate(double xoffset, double yoffset) override;
    bool MouseButtonUpdate(int button, int action, int mods) override;
    void Animate(float elapsedTimeSeconds) override;
    void Render(nvrhi::IFramebuffer *framebuffer) override;
    void DisplayScaleChanged(float scaleX, float scaleY) override;

 protected:
    virtual void BuildUI() = 0;

    void BeginFullScreenWindow();
    void DrawScreenCenteredText(const char *text);
    void EndFullScreenWindow();

    ImGuiContext *m_imContext = nullptr;
    nvrhi::CommandListHandle m_commandList;
    engine::ShaderFactory* m_shaderFactory = nullptr;
};

} // namespace donut::app

#endif /* IMGUILAYER_H */
