# Sandbox Sidecars

<Callout variant="beta">

There are currently several [known limitations](#limitations).

</Callout>

## Introduction

Sandbox Sidecars let you run additional containers alongside your main
Sandbox container, on the same host. A sandbox and its sidecars are connected
via an internal bridge network, allowing low latency communication between
containers over TCP/UDP, making them ideal for:

* Separating an agent harness from its execution environment, by running the
  agent in one container and its tool calls in another. See the [agent example](/docs/examples/sidecar_agent)
  for the general shape.
* Credentials injection, by running a proxy in a separate, trusted container
  from the primary application, and letting that proxy inject credentials
  or other secrets before passing on network calls to external services. The
  [credential proxy example](/docs/examples/sidecar_secrets_injection) uses
  a Modal [Secret](/docs/guide/secrets) that is injected only in the sidecar.
* Splitting out complex multi-service applications over separate containers,
  such as databases, caches or worker processes, similar to Docker Compose. The
  [Redis example](/docs/examples/sidecar_redis) shows the pattern.

We're still discovering all the ways that Sandbox Sidecars can be used - if you
come up with another use case, please let us know!

Sidecars are managed through the sidecars interface on a Sandbox
(`_experimental_sidecars` in Python, `experimentalSidecars` in JS/Go),
which provides methods to create, list, get, and terminate Sidecar containers.

Each Sidecar container:

* Runs its own image independently from the main Sandbox container.
* Runs in a separate, sandboxed process isolated from the main Sandbox container and other Sidecar containers.
* Can communicate over an internal bridge network with the main Sandbox container and other Sidecar containers.
* Can be created, terminated, and replaced dynamically during the Sandbox's lifetime.
* Supports executing commands just like the main Sandbox container.

## Usage

### Creating a Sidecar container

The main Sandbox container is resolvable as `main`, and each Sidecar container
is resolvable by the `name` you give it at creation time. Names are resolved using `/etc/hosts` which gets
updated when a sidecar is created or terminated.

<CodeTabs>
{#snippet python()}

```python notest
import modal

app = modal.App.lookup("sidecar-example", create_if_missing=True)
image = modal.Image.debian_slim().build(app)

sb = modal.Sandbox.create("sleep", "600", app=app, image=image, timeout=300)

sidecar = sb._experimental_sidecars.create(
    "python",
    "-m",
    "http.server",
    "8080",
    name="web",
    image=image,
)

# Give the server a moment to start, then call it from the main sandbox.
p = sb.exec(
    "python",
    "-c",
    "import time, urllib.request; time.sleep(1); print(urllib.request.urlopen('http://web:8080').status)",
)
p.wait()
print(p.stdout.read())  # "200"

sb.terminate()
```

{/snippet}

{#snippet javascript()}

```javascript notest
import { ModalClient } from "modal";

const modal = new ModalClient();
const app = await modal.apps.fromName("sidecar-example", {
  createIfMissing: true,
});
const image = await modal.images.fromRegistry("python:3.13-slim").build(app);

const sb = await modal.sandboxes.create(app, image, {
  command: ["sleep", "600"],
  timeoutMs: 300 * 1000,
});

const sidecar = await sb.experimentalSidecars.create("web", image, {
  command: ["python", "-m", "http.server", "8080"],
});

// Give the server a moment to start, then call it from the main sandbox.
const p = await sb.exec([
  "python",
  "-c",
  "import time, urllib.request; time.sleep(1); print(urllib.request.urlopen('http://web:8080').status)",
]);
await p.wait();
console.log(await p.stdout.readText()); // "200"

await sb.terminate();
```

{/snippet}

{#snippet go()}

```go
package main

import (
	"context"
	"fmt"
	"io"
	"time"

	modal "github.com/modal-labs/modal-client/go"
)

func main() {
	ctx := context.Background()
	mc, _ := modal.NewClient()

	app, _ := mc.Apps.FromName(ctx, "sidecar-example", &modal.AppFromNameParams{
		CreateIfMissing: true,
	})
	image, _ := mc.Images.FromRegistry("python:3.13-slim", nil).Build(ctx, app, nil)

	sb, _ := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
		Command: []string{"sleep", "600"},
		Timeout: 5 * time.Minute,
	})
	defer sb.Terminate(ctx, nil)

	sidecar, _ := sb.ExperimentalSidecars.Create(ctx, "web", image, &modal.SidecarCreateParams{
		Command: []string{"python", "-m", "http.server", "8080"},
	})
	_ = sidecar

	// Give the server a moment to start, then call it from the main sandbox.
	p, _ := sb.Exec(ctx, []string{
		"python", "-c",
		"import time, urllib.request; time.sleep(1); print(urllib.request.urlopen('http://web:8080').status)",
	}, nil)
	stdout, _ := io.ReadAll(p.Stdout)
	fmt.Println(string(stdout)) // "200"
}
```

{/snippet} </CodeTabs>

Sidecars use the same [runtime](/docs/guide/sandboxes#runtimes) as the main Sandbox.

### Listing and retrieving sidecars

You can list all running Sidecar containers or retrieve a specific one by name:

<CodeTabs>
{#snippet python()}

```python notest
containers = sb._experimental_sidecars.list()
for container in containers:
    print(f"{container.name}: {container.object_id}")

sidecar = sb._experimental_sidecars.get(name="web")
```

{/snippet}

{#snippet javascript()}

```javascript notest
const containers = await sb.experimentalSidecars.list();
for (const container of containers) {
  console.log(`${container.containerName}: ${container.containerId}`);
}

const sidecar = await sb.experimentalSidecars.get("web");
```

{/snippet}

{#snippet go()}

```go notest
containers, _ := sb.ExperimentalSidecars.List(ctx, nil)
for _, container := range containers {
	fmt.Printf("%s: %s\n", container.ContainerName, container.ContainerID)
}

sidecar, _ := sb.ExperimentalSidecars.Get(ctx, "web", nil)
_ = sidecar
```

{/snippet} </CodeTabs>

### Routing HTTPS traffic through a Sidecar

Sidecars can be used to inspect the outgoing HTTPS traffic from the main Sandbox
container in a different context, for example to perform more advanced request
filtering, inspecting requests for logging, or injecting secrets that the main
Sandbox container should not have access to.

Normally, applications need to support explicit proxy configuration, such as
respecting the `HTTPS_PROXY` environment variable, to route traffic through a Sidecar.
To include HTTPS traffic (TCP on port 443) from **proxy-unaware** applications, you can
set the experimental `proxy_traffic_via_sidecar` option that routes **all** outbound
HTTPS traffic from the main Sandbox container through a Sidecar.

<CodeTabs>
{#snippet python()}

```python notest
sb = modal.Sandbox.create(
    "sleep",
    "600",
    app=app,
    image=image,
    experimental_options={"proxy_traffic_via_sidecar": "my-proxy-sidecar"},
)

# Until this Sidecar is running, HTTPS from the main container is refused.
sb._experimental_sidecars.create("python", "/proxy.py", name="my-proxy-sidecar", image=proxy_image)
```

{/snippet}

{#snippet javascript()}

```javascript notest
const sb = await modal.sandboxes.create(app, image, {
  command: ["sleep", "600"],
  experimentalOptions: { proxy_traffic_via_sidecar: "my-proxy-sidecar" },
});

// Until this Sidecar is running, HTTPS from the main container is refused.
await sb.experimentalSidecars.create("my-proxy-sidecar", proxyImage, {
  command: ["python", "/proxy.py"],
});
```

{/snippet}

{#snippet go()}

```go notest
sb, _ := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
	Command:             []string{"sleep", "600"},
	ExperimentalOptions: map[string]any{"proxy_traffic_via_sidecar": "my-proxy-sidecar"},
})
// Until this Sidecar is running, HTTPS from the main container is refused.
sb.ExperimentalSidecars.Create(ctx, "my-proxy-sidecar", proxyImage, &modal.SidecarCreateParams{
	Command: []string{"python", "/proxy.py"},
})
```

{/snippet} </CodeTabs>

The Sidecar receives a raw TLS stream and must read the destination hostname
from the `ClientHello`'s SNI. The original destination IP is not forwarded, so
mechanisms such as `SO_ORIGINAL_DST` do not work. To read or rewrite HTTP
requests, terminate TLS in the proxy using a certificate authority that the
Sandbox trusts. See the [Sidecar traffic routing
example](/docs/examples/sidecar_traffic_routing) for a complete mitmproxy-based
request filter.

Only TCP traffic to port 443 is relayed. The Sandbox's own egress controls are
set aside for it: an `outbound_cidr_allowlist` on the Sandbox still governs every
other port, but relayed traffic passes regardless of what it lists. Non-relayed
traffic is still subject to the egress controls of the Sandbox.

Relayed traffic is instead governed by the egress controls of the Sidecar it is
relayed into. A Sidecar's outbound network policy is independent of the main
container's and defaults to open, so unless you pass `outbound_cidr_allowlist`
or `outbound_domain_allowlist` to the Sidecar itself, relayed traffic reaches
any destination the Sidecar chooses to connect to.

The option cannot be combined with setting `block_network`,
`outbound_domain_allowlist` or `proxy` on the Sandbox.

### Modal Proxy

You can specify that traffic from a Sidecar should exit through a
[Proxy](/docs/guide/proxy-ips) by passing a `proxy` at Sidecar creation, just as
for a Sandbox.

### OIDC tokens

Like Sandboxes, Sidecars do not receive an [OIDC](/docs/guide/oidc-integration)
token by default. To opt in, pass `include_oidc_identity_token=True` when creating
a Sidecar. The token is then available inside the Sidecar via the
`MODAL_IDENTITY_TOKEN` environment variable. The Sidecar’s identity token is issued
for its container, so its `container_id` claim differs from the corresponding Sandbox’s.

### Filesystem snapshots

You can snapshot a running Sidecar's filesystem into a reusable Image. The
resulting Image can be used anywhere an existing Image is accepted; the example
below uses it to start another Sidecar. The snapshot is scoped to that Sidecar;
it does not include the main Sandbox filesystem or other Sidecars.

<CodeTabs>
{#snippet python()}

```python notest
sidecar.filesystem.write_text("ready", "/tmp/state")
snapshot = sidecar.snapshot_filesystem()

restored = sb._experimental_sidecars.create(
    "sleep", "600", name="restored", image=snapshot
)
assert restored.filesystem.read_text("/tmp/state") == "ready"
```

{/snippet}

{#snippet javascript()}

```javascript notest
await sidecar.filesystem.writeText("ready", "/tmp/state");
const snapshot = await sidecar.snapshotFilesystem();

const restored = await sb.experimentalSidecars.create("restored", snapshot, {
  command: ["sleep", "600"],
});
console.assert((await restored.filesystem.readText("/tmp/state")) === "ready");
```

{/snippet}

{#snippet go()}

```go notest
_ = sidecar.Filesystem.WriteText(ctx, "ready", "/tmp/state", nil)
snapshot, _ := sidecar.SnapshotFilesystem(ctx, nil)

restored, _ := sb.ExperimentalSidecars.Create(ctx, "restored", snapshot, &modal.SidecarCreateParams{
	Command: []string{"sleep", "600"},
})
state, _ := restored.Filesystem.ReadText(ctx, "/tmp/state", nil)
fmt.Println(state) // "ready"
```

{/snippet} </CodeTabs>

### Directory mounts and snapshots

You can mount an Image at an absolute path in a running Sidecar and unmount it
later. You can also [snapshot a Sidecar directory](/docs/guide/sandbox-snapshots#directory-snapshots) into a new Image. The resulting
Image can be used anywhere an existing Image is accepted, including as a mount
or as the filesystem for a new container.

The example below uses a mounted `/workspace` as session state, snapshots it,
terminates the original Sidecar, and mounts the snapshot in a replacement
Sidecar:

<CodeTabs>
{#snippet python()}

```python notest
sidecar.mount_image("/workspace", modal.Image.from_scratch())
sidecar.filesystem.write_text("ready", "/workspace/state")

workspace = sidecar.snapshot_directory("/workspace")
sidecar.terminate(wait=True)

replacement = sb._experimental_sidecars.create(
    "sleep", "600", name="replacement", image=image
)
replacement.mount_image("/workspace", workspace)
assert replacement.filesystem.read_text("/workspace/state") == "ready"
replacement.unmount_image("/workspace")
```

{/snippet}

{#snippet javascript()}

```javascript notest
await sidecar.mountImage("/workspace");
await sidecar.filesystem.writeText("ready", "/workspace/state");

const workspace = await sidecar.snapshotDirectory("/workspace");
await sidecar.terminate({ wait: true });

const replacement = await sb.experimentalSidecars.create("replacement", image, {
  command: ["sleep", "600"],
});
await replacement.mountImage("/workspace", workspace);
console.assert(
  (await replacement.filesystem.readText("/workspace/state")) === "ready",
);
await replacement.unmountImage("/workspace");
```

{/snippet}

{#snippet go()}

```go notest
_ = sidecar.MountImage(ctx, "/workspace", nil, nil)
_ = sidecar.Filesystem.WriteText(ctx, "ready", "/workspace/state", nil)

workspace, _ := sidecar.SnapshotDirectory(ctx, "/workspace", nil)
_, _ = sidecar.Terminate(ctx, &modal.SidecarTerminateParams{Wait: true})

replacement, _ := sb.ExperimentalSidecars.Create(ctx, "replacement", image, &modal.SidecarCreateParams{
	Command: []string{"sleep", "600"},
})
_ = replacement.MountImage(ctx, "/workspace", workspace, nil)
state, _ := replacement.Filesystem.ReadText(ctx, "/workspace/state", nil)
fmt.Println(state) // "ready"
_ = replacement.UnmountImage(ctx, "/workspace", nil)
```

{/snippet} </CodeTabs>

### Volumes

A Sidecar can mount [Volumes](/docs/guide/volumes), configured the same way as
on a Sandbox. Each container gets its own mount. You can mount the same Volume
in the main Sandbox and in other Sidecars to share data between them.

### Cloud Bucket Mounts

A Sidecar can mount [Cloud Bucket Mounts](/docs/guide/cloud-bucket-mounts),
configured the same way as on a Sandbox. Each container gets its own mount:
a bucket mounted in a Sidecar is not visible in the main Sandbox container or
in other Sidecars, so mount it in every container that needs it. Cloud Bucket
Mounts are not supported in Sidecars of GPU Sandboxes.

<CodeTabs>
{#snippet python()}

```python notest
bucket = modal.CloudBucketMount(
    "my-bucket",
    secret=modal.Secret.from_name("my-aws-secret"),
    read_only=True,
)

reader = sb._experimental_sidecars.create(
    "sleep", "600", name="reader", image=image, volumes={"/mnt/bucket": bucket}
)
p = reader.exec("ls", "/mnt/bucket")
p.wait()
print(p.stdout.read())
```

{/snippet}

{#snippet javascript()}

```javascript notest
const secret = await modal.secrets.fromName("my-aws-secret");

const reader = await sb.experimentalSidecars.create("reader", image, {
  command: ["sleep", "600"],
  cloudBucketMounts: {
    "/mnt/bucket": modal.cloudBucketMounts.create("my-bucket", {
      secret,
      readOnly: true,
    }),
  },
});
const p = await reader.exec(["ls", "/mnt/bucket"]);
await p.wait();
console.log(await p.stdout.readText());
```

{/snippet}

{#snippet go()}

```go notest
secret, _ := mc.Secrets.FromName(ctx, "my-aws-secret", nil)
bucket, _ := mc.CloudBucketMounts.New("my-bucket", &modal.CloudBucketMountParams{
	Secret:   secret,
	ReadOnly: true,
})

reader, _ := sb.ExperimentalSidecars.Create(ctx, "reader", image, &modal.SidecarCreateParams{
	Command:           []string{"sleep", "600"},
	CloudBucketMounts: map[string]*modal.CloudBucketMount{"/mnt/bucket": bucket},
})
p, _ := reader.Exec(ctx, []string{"ls", "/mnt/bucket"}, nil)
stdout, _ := io.ReadAll(p.Stdout)
fmt.Println(string(stdout))
```

{/snippet} </CodeTabs>

With [OIDC authentication](/docs/guide/cloud-bucket-mounts#using-oidc-identity-tokens),
the identity token is issued for the Sidecar container, so its `container_id`
claim differs from the main container's. An IAM trust policy that matches on
the full subject must allow the Sidecar's container ID as well.

## Resource configuration

The main Sandbox container and the Sidecar containers share the resource allocation (CPU and memory) of the Sandbox,
and resources are configured only on the Sandbox. When planning your
resource allocation, make sure the Sandbox is configured with enough CPU
and memory for all containers combined.

For example, if you want to run a Sandbox with two Sidecars, and you expect the main
container to use 1 CPU core and 512 MiB of memory, Sidecar A to use 0.5 CPU and 256 MiB,
and Sidecar B to use 0.5 CPU and 256 MiB, you should set the Sandbox's resources to at
least 2 CPUs and 1024 MiB to accommodate all three containers.

There is a hard limit of **250** concurrent sidecar containers per sandbox,
regardless of the resource reservation.

On the gVisor runtime, bursting is still possible; see the [guide to Sandbox resources and
pricing](/docs/guide/sandbox-resources) for more details.

On the VM runtime, Sidecars cannot be combined with memory bursting. Instead, you need to
reserve memory for any Sidecars when you launch the Sandbox. That reserve is incompatible with VM memory bursting,
so the Sandbox must set an equal memory request and limit. With every Sidecar launch, you either specify how much of
the reserve to use for that Sidecar, or leave the option empty to use the entire remaining reserve:

<CodeTabs>
{#snippet python()}

```python notest
sb = modal.Sandbox.create(
    "sleep",
    "600",
    app=app,
    image=image,
    timeout=300,
    runtime="vm",
    memory=(8192, 8192),
    experimental_options={"vm_sidecar_memory_reserve_mib": 1024},
)

sidecar = sb._experimental_sidecars.create(
    "python",
    "-m",
    "http.server",
    "8080",
    name="web",
    image=image,
    experimental_memory_reserve_consume_mib=512,
)

# Omit experimental_memory_reserve_consume_mib to give this sidecar
# the remaining reserve (512 MiB here).
worker = sb._experimental_sidecars.create(
    "sleep", "600", name="worker", image=image
)
```

{/snippet}

{#snippet javascript()}

```javascript notest
const sb = await modal.sandboxes.create(app, image, {
  command: ["sleep", "600"],
  timeoutMs: 300 * 1000,
  runtime: "vm",
  memoryMiB: 8192,
  memoryLimitMiB: 8192,
  experimentalOptions: { vm_sidecar_memory_reserve_mib: "1024" },
});

const sidecar = await sb.experimentalSidecars.create("web", image, {
  command: ["python", "-m", "http.server", "8080"],
  experimentalMemoryReserveConsumeMiB: 512,
});

// Omit experimentalMemoryReserveConsumeMiB to give this sidecar
// the remaining reserve (512 MiB here).
const worker = await sb.experimentalSidecars.create("worker", image, {
  command: ["sleep", "600"],
});
```

{/snippet}

{#snippet go()}

```go notest
sb, _ := mc.Sandboxes.Create(ctx, app, image, &modal.SandboxCreateParams{
	Command:             []string{"sleep", "600"},
	Timeout:             5 * time.Minute,
	Runtime:             modal.SandboxRuntimeVM,
	MemoryMiB:           8192,
	MemoryLimitMiB:      8192,
	ExperimentalOptions: map[string]any{"vm_sidecar_memory_reserve_mib": "1024"},
})

sidecar, _ := sb.ExperimentalSidecars.Create(ctx, "web", image, &modal.SidecarCreateParams{
	Command:                             []string{"python", "-m", "http.server", "8080"},
	ExperimentalMemoryReserveConsumeMiB: 512,
})
_ = sidecar

// Omit ExperimentalMemoryReserveConsumeMiB to give this sidecar
// the remaining reserve (512 MiB here).
worker, _ := sb.ExperimentalSidecars.Create(ctx, "worker", image, &modal.SidecarCreateParams{
	Command: []string{"sleep", "600"},
})
_ = worker
```

{/snippet} </CodeTabs>

Terminated VM Sidecars return their reserve for reuse by new Sidecars.

## Limitations

The main sandbox supports the same features as a regular sandbox, but some features are not yet supported
for sidecars:

* **Pre-built images only**: Sidecar images must already exist: pre-built with `image.build()`, a
  [named image](/docs/guide/named-images) via `Image.from_name()`, an
  ID via `Image.from_id()`, or created from [filesystem](/docs/guide/sandbox-snapshots#filesystem-snapshots)/[directory](/docs/guide/sandbox-snapshots#directory-snapshots) snapshots. Lazy image
  building is not supported for sidecars. See also
  [Separating Image builds from Sandbox creation](/docs/guide/sandboxes#separating-image-builds-from-sandbox-creation).
* **No stdin/stdout**: A Sidecar's entrypoint does not expose stdin, stdout, or stderr streams.
* **No VM memory bursting**: Sidecars on the VM runtime cannot be combined with VM memory bursting. Set an equal memory request and limit, e.g. `memory=(8192, 8192)`.
* **Limitations on GPU Sandboxes**: You can attach Sidecars to Sandboxes with GPUs, but the Sidecar is CPU-only and cannot attach [Cloud Bucket Mounts](/docs/guide/cloud-bucket-mounts).
* **No memory snapshot support**: A Sidecar's [filesystem](/docs/guide/sandbox-snapshots#filesystem-snapshots) and [individual directories](/docs/guide/sandbox-snapshots#directory-snapshots) can be independently snapshotted, but Sidecar memory state can not be captured with a
  [memory snapshot](/docs/guide/sandbox-snapshots#memory-snapshots).
* **Changes to /etc/hosts are not preserved**: `/etc/hosts` is rewritten on sidecar create/terminate and user changes are not preserved.
