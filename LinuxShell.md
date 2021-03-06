# Linux 命令行与 shell 脚本编程大全

# 第一部分 Linux 命令行

## 第1章 初识 Linux Shell

Linux 可以划分为以下 4 部分：
- Linux 内核
- GNU 工具
- 图形化桌面环境
- 应用软件

内核有许多种类型，但是它们基本上可以分为两类：规模较大的一类称为单内核，规模较小的一类称为微内核；

单内核特点：程序偏大复杂，速度快；
微内核特点：一个非常小的程序，只能执行最基本的任务，为了执行其他功能，微内核要调用其他程序，这些程序称为服务器；可以定制化，易于维护；

大多数 Unix 系统使用某种类型的单内核，但一些 Unix（如 Macintosh Unix，OS X）使用的是微内核；

最初黑客选择Linux的原因：Dos太low，Macintosh太贵；现在还是如此。

Minix 产生的原因：AT&T修改了政策，不允许公司以外的人查看 Unix 源码；并且那时的 Unix 系统不能跑在 PC 机上；广大的开发人员希望对官方版本进行增强，而 Tanenbaum 检测 Minix 必须足够简单，适用于教学；

Unix 由 AT&T 开发，由于AT&T对UNIX的商业化使用引起了伯克利分校的不满，他们决定重新开发受证书影响的组件，开发完成后命名为 386/BSD，之后该 UNIX 版本托管给一组志愿者进行管理，该小组将操作系统重新命名为 FreeBSD。刚开始FreeBSD只能在PC机上运行，有些用户希望在其他类型的机器上也能运行BSD，就成立了一个新的小组，目标就是将 FreeBSD 移植到许多其他类型的计算机上。这个新小组提供的版本称为 NetBSD。20世纪90年代中期，NetBSD小组又分出了另一个小组，该小组主要关注安全和密码学问题，这个小组的操作系统称为 OpenBSD。BSD 世界只有3个主要的发布版本：FreeBSD/NetBSD/OpenBSD。

BSD 许可证远没有 GPL 严格，在BSD许可证下，允许使用部分BSD创建新产品而不共享该新产品。当这种情况发生时，全世界很大程度上无法从新产品上获取好处，也不能使用和修改新产品。基于这一原因，许多程序员喜欢使用 Linux。

每个人类使用的机器都可以分为两个部分：界面和其他部件。在Unix中，我们称界面为终端(terminal)，其他部件总称为主机(host)。因为终端提供界面，所以它有两个主要的任务：接受输入和生成输出。

Unix系统总是区分控制台和普通终端。如果你是一名系统管理员，有一些特定的事情只能在控制台上完成，不能通过远程终端访问系统。

