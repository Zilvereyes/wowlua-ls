use mlua::prelude::*;

/// Instruction limit per plugin load or `run()` invocation.
const INSTRUCTION_LIMIT: u32 = 1_000_000;
const INSTRUCTION_LIMIT_ERROR: &str = "plugin exceeded instruction limit";
const INSTRUCTION_BUDGET_REGISTRY_KEY: &str = "wowlua.plugin_instruction_budget";

const PROTECTED_CALL_GUARDS: &str = r#"
local budget, budget_error = ...
local raw_error = error
local raw_pcall = pcall
local raw_xpcall = xpcall
local raw_select = select
local raw_unpack = unpack

local function pack(...)
    return { n = raw_select("#", ...), ... }
end

local function guarded_result(...)
    local result = pack(...)
    if budget.exhausted then
        raw_error(budget_error, 0)
    end
    return raw_unpack(result, 1, result.n)
end

pcall = function(...)
    return guarded_result(raw_pcall(...))
end

xpcall = function(fn, handler)
    local function guarded_handler(err)
        if budget.exhausted then
            return budget_error
        end
        return handler(err)
    end
    return guarded_result(raw_xpcall(fn, guarded_handler))
end
"#;

/// Create a sandboxed Lua 5.1 VM with restricted stdlib.
///
/// Available: string, table, math, pairs, ipairs, next, type, tostring, tonumber,
/// select, unpack, guarded pcall/xpcall, error, assert, rawequal, rawget, rawset.
///
/// Removed: os, io, debug, coroutine, loadfile, dofile, require, load, collectgarbage.
/// `print` is mapped to `log::info!`.
pub(super) fn create_sandbox() -> Lua {
    let lua = Lua::new();

    // Remove dangerous globals
    {
        let globals = lua.globals();
        for name in &[
            "os", "io", "debug", "coroutine", "loadfile", "dofile", "require",
            "load", "collectgarbage",
        ] {
            let _ = globals.set(*name, LuaNil);
        }

        // Replace print with a logging wrapper
        let print_fn = lua.create_function(|_, args: LuaMultiValue| {
            let parts: Vec<String> = args.into_iter().map(|v| format!("{v:?}")).collect();
            log::info!("[plugin] {}", parts.join("\t"));
            Ok(())
        }).expect("failed to create print wrapper");
        let _ = globals.set("print", print_fn);
    }

    install_protected_call_guards(&lua);

    // Set instruction count hook to prevent infinite loops. The private state is
    // captured by the guarded protected calls and cannot be cleared by plugins.
    // Re-arm it before each plugin invocation via `reset_instruction_limit()`.
    set_instruction_hook(&lua);

    lua
}

fn install_protected_call_guards(lua: &Lua) {
    let budget = lua.create_table().expect("failed to create instruction budget state");
    budget.raw_set("exhausted", false)
        .expect("failed to initialize instruction budget state");
    lua.set_named_registry_value(INSTRUCTION_BUDGET_REGISTRY_KEY, budget.clone())
        .expect("failed to store instruction budget state");

    lua.load(PROTECTED_CALL_GUARDS)
        .set_name("plugin protected-call guards")
        .call::<()>((budget, INSTRUCTION_LIMIT_ERROR))
        .expect("failed to install protected-call guards");
}

fn set_instruction_hook(lua: &Lua) {
    lua.set_hook(
        mlua::HookTriggers::new().every_nth_instruction(INSTRUCTION_LIMIT),
        |lua, _debug| {
            let budget: LuaTable =
                lua.named_registry_value(INSTRUCTION_BUDGET_REGISTRY_KEY)?;
            budget.raw_set("exhausted", true)?;
            Err(LuaError::runtime(INSTRUCTION_LIMIT_ERROR))
        },
    );
}

/// Reset the instruction counter before each plugin load or `run()` invocation.
/// The `every_nth_instruction` hook is cumulative, so without resetting
/// a heavy-but-legal plugin could starve subsequent plugins.
pub(super) fn reset_instruction_limit(lua: &Lua) {
    lua.remove_hook();
    let budget: LuaTable = lua.named_registry_value(INSTRUCTION_BUDGET_REGISTRY_KEY)
        .expect("instruction budget state should exist");
    budget.raw_set("exhausted", false)
        .expect("failed to reset instruction budget state");
    set_instruction_hook(lua);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sandbox_removes_dangerous_globals() {
        let lua = create_sandbox();
        let globals = lua.globals();
        for name in &[
            "os", "io", "debug", "coroutine", "loadfile", "dofile", "require",
            "load", "collectgarbage",
        ] {
            let val: LuaValue = globals.get(*name).unwrap();
            assert!(val.is_nil(), "{name} should be nil in sandbox");
        }
    }

    #[test]
    fn sandbox_keeps_safe_globals() {
        let lua = create_sandbox();
        let globals = lua.globals();
        for name in &[
            "string", "table", "math", "pairs", "ipairs", "type", "tostring",
            "tonumber", "pcall", "xpcall", "error", "assert",
        ] {
            let val: LuaValue = globals.get(*name).unwrap();
            assert!(!val.is_nil(), "{name} should exist in sandbox");
        }
    }

    #[test]
    fn sandbox_instruction_limit() {
        let lua = create_sandbox();
        let result = lua.load("while true do end").exec();
        assert!(result.is_err(), "infinite loop should hit instruction limit");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("instruction limit"), "error should mention instruction limit: {err}");
    }

    #[test]
    fn protected_calls_cannot_retry_after_instruction_limit() {
        for source in [
            "while true do pcall(function() while true do end end) end",
            "while true do xpcall(function() while true do end end, function() return 'caught' end) end",
            "xpcall(function() error('ordinary') end, function() while true do end end)",
        ] {
            let lua = create_sandbox();
            let result = lua.load(source).exec();
            assert!(result.is_err(), "protected loop should hit instruction limit");
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("instruction limit"),
                "protected error should preserve instruction limit: {err}"
            );
        }
    }

    #[test]
    fn protected_calls_preserve_ordinary_results() {
        let lua = create_sandbox();
        let pcall_result: (bool, i64, Option<i64>, i64) = lua
            .load("return pcall(function() return 1, nil, 3 end)")
            .eval()
            .unwrap();
        assert_eq!(pcall_result, (true, 1, None, 3));

        let xpcall_result: (bool, String) = lua
            .load(
                "return xpcall(function() error('ordinary') end, \
                 function(err) return 'handled: ' .. err end)",
            )
            .eval()
            .unwrap();
        assert!(!xpcall_result.0);
        assert!(xpcall_result.1.contains("handled: "));
        assert!(xpcall_result.1.contains("ordinary"));
    }
}
