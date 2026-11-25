// Scene-graph trace (aggregate docs/plans/2026-11-21-donut-ecs.md, step E0.3; docs/ecs-design.md 11.2).
//
// A seeded generator builds scenes through the public SceneGraph API: flat instances, a deep chain, a wide and deep
// tree, nodes without transforms, skinned characters (joint nodes with SkinnedMeshReference leaves, a
// SkinnedMeshInstance, an animation), lights and cameras. A per-frame script edits them: new transforms on a moving
// fraction of the nodes, animations, Detach and a later Attach of the detached subgraph, SetLeaf, AttachLeafNode, the
// deep copy of an attached subgraph, SetRootNode, and orphaned subgraphs built off the graph and attached in one piece.
// Every eighth frame changes nothing. After each Refresh the test records every node in pre-order, by path: the world
// transform (double bit patterns), the float world transform, the global bounding box, the leaf and subgraph content
// flags; then the order of every leaf vector, as the pre-order indices of the leaves' nodes, and the mesh and material
// counts. Step E3 replaces the node storage and must reproduce these records bit for bit.
//
// Only the API that design 5.7 keeps is used: no previous transforms, no dirty flags, no instance or geometry indices,
// no mesh or material ids. Getters are copied, never held by reference (they return by value after E3).
// Determinism: no std:: random distributions (their output is implementation-defined); every random value is drawn in
// its own statement (argument evaluation order is unspecified); no transcendental functions (their results may differ
// between Debug and the /arch:AVX2 Release build), so transforms use + - * / and sqrt only, and the animations
// interpolate linearly.
//
// Golden files, in DONUT_TEST_GOLDEN_DIR (set in tests/test-engine.cmake):
//     scene_graph_trace_hashes.txt     FNV-1a 64 of the record of every frame of every scene
//     scene_graph_trace_<scene>.txt    the full records of the first frames of each small scene
// Environment:
//     ECS_UPDATE_GOLDEN=1              rewrites the golden files instead of comparing them
//     ECS_TRACE_DUMP_DIR=<dir>         also writes the records of every frame of every scene to <dir>/<scene>.txt
//     ECS_TRACE_VERBOSE=1              prints the edits of every frame
// On a mismatch the test prints the first difference and writes the actual file to DONUT_TEST_BINARY_DIR.

#include "SceneGraphGenerator.h"

#include <donut/engine/SceneGraph.h>
#include <donut/core/query_cast.h>
#include <donut/tests/utils.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifndef DONUT_TEST_GOLDEN_DIR
#error "DONUT_TEST_GOLDEN_DIR is defined for this test in tests/test-engine.cmake"
#endif

using namespace donut;
using namespace donut::engine;

static_assert(sizeof(dm::daffine3) == 12 * sizeof(double), "the record reads a daffine3 as 12 doubles");
static_assert(sizeof(dm::affine3) == 12 * sizeof(float), "the record reads an affine3 as 12 floats");
static_assert(sizeof(dm::box3) == 6 * sizeof(float), "the record reads a box3 as 6 floats");

namespace
{
    // ------------------------------------------------------------------------------------------ records

    void AppendHex(std::string& out, uint64_t value, int digits)
    {
        static const char c_Digits[] = "0123456789abcdef";
        for (int i = digits - 1; i >= 0; --i)
            out += c_Digits[(value >> (4 * i)) & 0xf];
    }

    void AppendFlags(std::string& out, SceneContentFlags flags)
    {
        char text[16];
        std::snprintf(text, sizeof(text), "%x", uint32_t(flags));
        out += text;
    }

    std::string LeafTag(SceneGraphLeaf* leaf)
    {
        if (!leaf)
            return "-";
        if (auto skinned = query_cast<SkinnedMeshInstance>(leaf))
            return "K:" + skinned->GetPrototypeMesh()->name;
        if (auto mesh = query_cast<MeshInstance>(leaf))
            return "M:" + mesh->GetMesh()->name;
        if (query_cast<SkinnedMeshReference>(leaf))
            return "J";
        if (query_cast<DirectionalLight>(leaf))
            return "Ld";
        if (query_cast<PointLight>(leaf))
            return "Lp";
        if (query_cast<SpotLight>(leaf))
            return "Ls";
        if (query_cast<PerspectiveCamera>(leaf))
            return "Cp";
        if (query_cast<OrthographicCamera>(leaf))
            return "Co";
        if (query_cast<SceneGraphAnimation>(leaf))
            return "A";
        return "?";
    }

    // One frame: a line per node in pre-order, then the leaf vectors and the counts. Every `pathStride`-th node
    // also checks that GetPath agrees with the path the walk built.
    std::string RecordFrame(SceneGraph* graph, uint32_t frame, int pathStride)
    {
        std::string out = "frame " + std::to_string(frame) + "\n";
        std::unordered_map<const SceneGraphNode*, int> order;

        std::string path;
        std::vector<size_t> pathLengths;
        auto push = [&](const std::string& name) { pathLengths.push_back(path.size()); path += '/'; path += name; };
        auto pop = [&]() { path.resize(pathLengths.back()); pathLengths.pop_back(); };

        int index = 0;
        SceneGraphWalker walker(graph->GetRootNode());
        if (walker)
            push(walker->GetName());
        while (walker)
        {
            SceneGraphNode* node = walker.Get();
            order.emplace(node, index);
            if (index % pathStride == 0)
                CHECK(node->GetPath().generic_string() == path);

            const dm::daffine3 world = node->GetLocalToWorldTransform();
            const dm::affine3 worldFloat = node->GetLocalToWorldTransformFloat();
            const dm::box3 bounds = node->GetGlobalBoundingBox();
            const SceneContentFlags leafFlags = node->GetLeafContentFlags();
            const SceneContentFlags subgraphFlags = node->GetSubgraphContentFlags();

            uint64_t worldBits[12];
            uint32_t floatBits[12];
            uint32_t boundsBits[6];
            std::memcpy(worldBits, &world, sizeof(worldBits));
            std::memcpy(floatBits, &worldFloat, sizeof(floatBits));
            std::memcpy(boundsBits, &bounds, sizeof(boundsBits));

            out += path;
            out += ' ';
            out += LeafTag(node->GetLeaf());
            out += " L";
            AppendFlags(out, leafFlags);
            out += " S";
            AppendFlags(out, subgraphFlags);
            out += " W";
            for (uint64_t bits : worldBits)
            {
                out += ' ';
                AppendHex(out, bits, 16);
            }
            out += " F";
            for (uint32_t bits : floatBits)
            {
                out += ' ';
                AppendHex(out, bits, 8);
            }
            out += " B";
            for (uint32_t bits : boundsBits)
            {
                out += ' ';
                AppendHex(out, bits, 8);
            }
            out += '\n';

            ++index;
            const int delta = walker.Next(true);
            if (!walker)
                break;
            for (int i = delta; i <= 0; ++i)   // a child pops nothing, a sibling pops one, going up pops more
                pop();
            push(walker->GetName());
        }

        auto appendLeaves = [&](const char* label, const auto& leaves)
        {
            out += label;
            for (const auto& leaf : leaves)
            {
                const auto it = order.find(leaf->GetNode());
                out += ' ';
                out += std::to_string(it != order.end() ? it->second : -1);
            }
            out += '\n';
        };
        appendLeaves("mesh_instances", graph->GetMeshInstances());
        appendLeaves("skinned_mesh_instances", graph->GetSkinnedMeshInstances());
        appendLeaves("lights", graph->GetLights());
        appendLeaves("cameras", graph->GetCameras());
        appendLeaves("animations", graph->GetAnimations());

        out += "nodes " + std::to_string(index) + " meshes " + std::to_string(graph->GetMeshes().size())
            + " materials " + std::to_string(graph->GetMaterials().size()) + "\n";
        return out;
    }

    uint64_t Fnv1a(const std::string& text)
    {
        uint64_t hash = 0xcbf29ce484222325ull;
        for (unsigned char c : text)
        {
            hash ^= c;
            hash *= 0x100000001b3ull;
        }
        return hash;
    }

    // ------------------------------------------------------------------------------------------ runs

    struct SceneSpec
    {
        const char* name;
        uint64_t seed;
        const Profile* profile;
        void (*build)(Scene&, uint32_t);
        uint32_t size;
        uint32_t frames;
        uint32_t fullFrames;   // the first frames whose full records are golden files (small scenes only)
    };

    const SceneSpec c_Scenes[] = {
        { "flat_small",    0x1001, &c_SmallProfile, BuildFlat,    12, 40, 3 },
        { "chain_small",   0x1002, &c_SmallProfile, BuildChain,   10, 40, 3 },
        { "tree_small",    0x1003, &c_SmallProfile, BuildTree,    20, 40, 3 },
        { "skinned_small", 0x1004, &c_SmallProfile, BuildCrowd,    1, 40, 3 },
        { "mixed_small",   0x1005, &c_SmallProfile, BuildMixed,   24, 40, 3 },
        { "flat",          0x2001, &c_LargeProfile, BuildFlat,  4000, 40, 0 },
        { "chain",         0x2002, &c_LargeProfile, BuildChain,  300, 40, 0 },
        { "tree",          0x2003, &c_LargeProfile, BuildTree,  3000, 40, 0 },
        { "skinned",       0x2004, &c_LargeProfile, BuildCrowd,   40, 40, 0 },
        { "mixed",         0x2005, &c_LargeProfile, BuildMixed, 1200, 40, 0 },
    };

    struct Trace
    {
        std::string hashes;       // "<scene> <frame> <hash>" lines
        std::string fullRecords;  // the first fullFrames records
        size_t nodesAtEnd = 0;
    };

    Trace RunScene(const SceneSpec& spec, std::string* everyRecord)
    {
        if (g_Verbose)
            std::printf("%s\n", spec.name);
        Scene s(spec.seed, *spec.profile);
        spec.build(s, spec.size);
        s.nodeLimit = CollectNodes(s.graph).size() * 3 / 2 + 64;
        const int pathStride = spec.fullFrames ? 1 : 61;

        Trace trace;
        for (uint32_t frame = 0; frame < spec.frames; ++frame)
        {
            if (frame > 0)
                Step(s, frame);
            s.graph->Refresh(frame);

            const std::string record = RecordFrame(s.graph, frame, pathStride);
            trace.hashes += std::string(spec.name) + ' ' + std::to_string(frame) + ' ';
            AppendHex(trace.hashes, Fnv1a(record), 16);
            trace.hashes += '\n';
            if (frame < spec.fullFrames)
                trace.fullRecords += record;
            if (everyRecord)
                *everyRecord += record;
        }
        trace.nodesAtEnd = CollectNodes(s.graph).size();
        return trace;
    }

    // ------------------------------------------------------------------------------------------ golden files

    bool ReadFile(const std::filesystem::path& path, std::string& text)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.good())
            return false;
        std::stringstream buffer;
        buffer << file.rdbuf();
        text = buffer.str();
        text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
        return true;
    }

    void WriteFile(const std::filesystem::path& path, const std::string& text)
    {
        std::ofstream file(path, std::ios::binary);
        file << text;
        CHECK(file.good());
    }

    std::vector<std::string_view> Split(std::string_view text, char separator)
    {
        std::vector<std::string_view> parts;
        size_t start = 0;
        while (start <= text.size())
        {
            const size_t end = std::min(text.find(separator, start), text.size());
            parts.push_back(text.substr(start, end - start));
            start = end + 1;
        }
        return parts;
    }

    // A record token as a number: 16 hex digits are a double, 8 hex digits a float.
    std::string Decode(std::string_view token)
    {
        const bool hex = !token.empty() && token.find_first_not_of("0123456789abcdef") == std::string_view::npos;
        char text[64] = "";
        if (hex && token.size() == 16)
        {
            const uint64_t bits = std::strtoull(std::string(token).c_str(), nullptr, 16);
            double value;
            std::memcpy(&value, &bits, sizeof(value));
            std::snprintf(text, sizeof(text), " (%.17g)", value);
        }
        else if (hex && token.size() == 8)
        {
            const uint32_t bits = uint32_t(std::strtoul(std::string(token).c_str(), nullptr, 16));
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            std::snprintf(text, sizeof(text), " (%.9g)", double(value));
        }
        return text;
    }

    void ReportFirstDifference(const std::string& name, const std::string& expected, const std::string& actual)
    {
        const auto expectedLines = Split(expected, '\n');
        const auto actualLines = Split(actual, '\n');
        std::string_view frame;
        for (size_t line = 0; line < std::max(expectedLines.size(), actualLines.size()); ++line)
        {
            const std::string_view e = line < expectedLines.size() ? expectedLines[line] : "<end of file>";
            const std::string_view a = line < actualLines.size() ? actualLines[line] : "<end of file>";
            if (e == a)
            {
                if (e.substr(0, 6) == "frame ")
                    frame = e;
                continue;
            }

            std::printf("  first difference in %s at line %zu%s%.*s:\n", name.c_str(), line + 1, frame.empty() ? "" : ", in ",
                int(frame.size()), frame.data());
            std::printf("    golden: %.*s\n", int(e.size()), e.data());
            std::printf("    actual: %.*s\n", int(a.size()), a.data());

            const auto expectedTokens = Split(e, ' ');
            const auto actualTokens = Split(a, ' ');
            std::string_view field;
            int element = -1;
            for (size_t t = 0; t < std::min(expectedTokens.size(), actualTokens.size()); ++t)
            {
                if (expectedTokens[t] == "W" || expectedTokens[t] == "F" || expectedTokens[t] == "B")
                {
                    field = expectedTokens[t];
                    element = -1;
                    continue;
                }
                ++element;
                if (expectedTokens[t] == actualTokens[t])
                    continue;
                if (!field.empty())
                    std::printf("    %.*s element %d: golden %.*s%s, actual %.*s%s\n", int(field.size()), field.data(), element,
                        int(expectedTokens[t].size()), expectedTokens[t].data(), Decode(expectedTokens[t]).c_str(),
                        int(actualTokens[t].size()), actualTokens[t].data(), Decode(actualTokens[t]).c_str());
                break;
            }
            return;
        }
    }

    bool CompareGolden(const std::string& name, const std::string& actual, bool update)
    {
        const std::filesystem::path path = std::filesystem::path(DONUT_TEST_GOLDEN_DIR) / name;
        if (update)
        {
            std::filesystem::create_directories(path.parent_path());
            WriteFile(path, actual);
            std::printf("  wrote %s (%zu bytes)\n", path.generic_string().c_str(), actual.size());
            return true;
        }

        std::string expected;
        if (!ReadFile(path, expected))
        {
            std::printf("  golden file missing: %s (run with ECS_UPDATE_GOLDEN=1)\n", path.generic_string().c_str());
            return false;
        }
        if (expected == actual)
            return true;

        ReportFirstDifference(name, expected, actual);
        const std::filesystem::path actualPath = std::filesystem::path(DONUT_TEST_BINARY_DIR) / (name + ".actual");
        WriteFile(actualPath, actual);
        std::printf("  the actual records are in %s\n", actualPath.generic_string().c_str());
        std::printf("  (the records of every frame: ECS_TRACE_DUMP_DIR=<dir>, on this build and on one that matches)\n");
        return false;
    }
}

int main(int, char**)
{
    const char* updateVariable = std::getenv("ECS_UPDATE_GOLDEN");
    const bool update = updateVariable && std::strcmp(updateVariable, "1") == 0;
    const char* dumpDirectory = std::getenv("ECS_TRACE_DUMP_DIR");
    const char* verboseVariable = std::getenv("ECS_TRACE_VERBOSE");
    g_Verbose = verboseVariable && std::strcmp(verboseVariable, "1") == 0;

    try
    {
        bool ok = true;
        std::string hashes;
        for (const SceneSpec& spec : c_Scenes)
        {
            const auto start = std::chrono::steady_clock::now();
            std::string everyRecord;
            const Trace trace = RunScene(spec, dumpDirectory ? &everyRecord : nullptr);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::printf("%-14s %u frames, %zu nodes at the end, %.2f s\n", spec.name, spec.frames, trace.nodesAtEnd, seconds);

            if (spec.fullFrames)
            {
                // a second run in this process, with other heap addresses, records the same frames
                const Trace again = RunScene(spec, nullptr);
                CHECK(again.hashes == trace.hashes);
                CHECK(again.fullRecords == trace.fullRecords);
                ok = CompareGolden("scene_graph_trace_" + std::string(spec.name) + ".txt", trace.fullRecords, update) && ok;
            }
            if (dumpDirectory)
            {
                std::filesystem::create_directories(dumpDirectory);
                WriteFile(std::filesystem::path(dumpDirectory) / (std::string(spec.name) + ".txt"), everyRecord);
            }
            hashes += trace.hashes;
        }
        ok = CompareGolden("scene_graph_trace_hashes.txt", hashes, update) && ok;
        if (!ok)
        {
            std::fprintf(stderr, "the scene-graph trace differs from the golden files\n");
            return 1;
        }
    }
    catch (const std::runtime_error& err)
    {
        fprintf(stderr, "%s", err.what());
        return 1;
    }
    return 0;
}
